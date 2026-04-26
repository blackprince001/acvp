"""
Detection-model benchmark runner.

Per-model recipe:
  1. Load weights into Ultralytics YOLO/RTDETR.
  2. Run ``model.val(...)`` on the test split → mAP / P / R / per-class AP.
  3. Run a separate latency loop on real test images/video → latency distribution.
  4. Collect static info (param count, GFLOPs, weights MB).

Outputs a ``BenchmarkResult`` per model + writes them all to JSON/CSV in
``output_dir``. The runner intentionally evaluates the *model in isolation*
— no tracker, no spatial filter, no orchestrator overhead — so results
reflect the model itself, the way you'd report them in a paper.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from loguru import logger
from ultralytics import RTDETR, YOLO

from src.experiment.tracker import ExperimentTracker

from .metrics import DetectionMetrics, LatencyStats, ModelInfo


@dataclass
class BenchmarkResult:
  """Single-model bench record."""

  model_name: str
  weights_path: str
  device: str
  image_size: int
  data_yaml: str
  split: str
  detection: dict[str, Any]
  latency: dict[str, Any]
  info: dict[str, Any]
  timing: dict[str, float] = field(default_factory=dict)

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


@dataclass
class BenchmarkConfig:
  data_yaml: Path
  output_dir: Path
  split: str = "test"
  image_size: int = 640
  device: str = "cuda"
  batch: int = 1
  conf: float = 0.001  # standard val-time conf for mAP
  iou: float = 0.6
  latency_warmup: int = 10
  latency_iters: int = 100
  latency_max_images: int = 200
  latency_video: Path | None = None  # if set, latency loop reads frames from this video
  latency_video_stride: int = 1


class DetectionBenchmark:
  """Run ``model.val()`` + a latency loop for a single detection model."""

  def __init__(self, config: BenchmarkConfig, tracker: ExperimentTracker | None = None):
    self.config = config
    self.tracker = tracker or ExperimentTracker()
    self.config.output_dir.mkdir(parents=True, exist_ok=True)

  def run_one(self, model_name: str, weights: str | Path) -> BenchmarkResult:
    weights = str(weights)
    logger.info("[{}] loading weights from {}", model_name, weights)
    model = self._load(model_name, weights)

    info = ModelInfo.from_ultralytics(model_name, model.model, weights)
    logger.info(
      "[{}] params={:,} | weights={:.2f} MB | gflops={}",
      model_name,
      info.parameters,
      info.weights_mb,
      info.gflops,
    )

    # ---- accuracy ----
    t0 = time.perf_counter()
    val_results = model.val(
      data=str(self.config.data_yaml),
      split=self.config.split,
      imgsz=self.config.image_size,
      batch=self.config.batch,
      conf=self.config.conf,
      iou=self.config.iou,
      device=self.config.device,
      verbose=False,
      plots=False,
    )
    val_seconds = time.perf_counter() - t0
    class_names = list(model.names.values()) if hasattr(model, "names") else []
    det = DetectionMetrics.from_ultralytics(val_results, class_names)
    logger.info(
      "[{}] mAP@0.5={:.4f} mAP@0.5:0.95={:.4f} P={:.4f} R={:.4f}",
      model_name,
      det.map50,
      det.map50_95,
      det.precision,
      det.recall,
    )

    # ---- latency ----
    if self.config.latency_video is not None:
      frames = self._gather_video_frames()
      logger.info(
        "[{}] latency source: video {} ({} frames decoded)",
        model_name,
        self.config.latency_video,
        len(frames),
      )
    else:
      frames = self._gather_image_frames()
      logger.info(
        "[{}] latency source: test split images ({} frames)",
        model_name,
        len(frames),
      )
    lat = self._latency_loop(model, frames)
    logger.info(
      "[{}] latency mean={:.2f}ms p95={:.2f}ms fps={:.1f}",
      model_name,
      lat.mean_ms,
      lat.p95_ms,
      lat.fps_mean,
    )

    latency_dict = lat.to_dict()
    latency_dict["source"] = (
      f"video:{self.config.latency_video}"
      if self.config.latency_video is not None
      else f"split:{self.config.split}"
    )
    result = BenchmarkResult(
      model_name=model_name,
      weights_path=weights,
      device=self.config.device,
      image_size=self.config.image_size,
      data_yaml=str(self.config.data_yaml),
      split=self.config.split,
      detection=det.to_dict(),
      latency=latency_dict,
      info=info.to_dict(),
      timing={"val_seconds": round(val_seconds, 2)},
    )

    self._log_to_tracker(result)
    self._write_json(result)

    if self.config.device.startswith("cuda"):
      torch.cuda.empty_cache()
    return result

  def run_many(self, models: dict[str, str | Path]) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    for name, weights in models.items():
      try:
        results.append(self.run_one(name, weights))
      except Exception as exc:  # noqa: BLE001
        logger.exception("[{}] benchmark failed: {}", name, exc)
    self._write_summary(results)
    return results

  def _load(self, model_name: str, weights: str) -> YOLO | RTDETR:
    name = model_name.lower()
    cls = RTDETR if name.startswith("rtdetr") else YOLO
    model = cls(weights)
    return model

  def _gather_image_frames(self) -> list[np.ndarray]:
    """Decode test-split images, resized to the bench image size."""
    import yaml

    with open(self.config.data_yaml) as f:
      data = yaml.safe_load(f)
    base = Path(data.get("path", self.config.data_yaml.parent))
    rel = data.get(self.config.split)
    if rel is None:
      raise ValueError(f"{self.config.data_yaml} has no '{self.config.split}' entry")
    img_dir = (base / rel).resolve()
    if not img_dir.is_dir():
      raise FileNotFoundError(f"Image dir not found: {img_dir}")
    paths = sorted(
      [p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if not paths:
      raise RuntimeError(f"No images in {img_dir}")
    if len(paths) > self.config.latency_max_images:
      paths = paths[: self.config.latency_max_images]
    imgsz = self.config.image_size
    frames: list[np.ndarray] = []
    for p in paths:
      img = cv2.imread(str(p))
      if img is None:
        continue
      frames.append(cv2.resize(img, (imgsz, imgsz)))
    if not frames:
      raise RuntimeError("No readable images for latency loop.")
    return frames

  def _gather_video_frames(self) -> list[np.ndarray]:
    """Decode up to ``latency_max_images`` frames from the configured video."""
    video_path = self.config.latency_video
    assert video_path is not None
    if not Path(video_path).exists():
      raise FileNotFoundError(f"Latency video not found: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
      raise RuntimeError(f"Could not open video: {video_path}")
    imgsz = self.config.image_size
    stride = max(1, self.config.latency_video_stride)
    frames: list[np.ndarray] = []
    idx = 0
    try:
      while len(frames) < self.config.latency_max_images:
        ok, frame = cap.read()
        if not ok:
          break
        if idx % stride == 0:
          frames.append(cv2.resize(frame, (imgsz, imgsz)))
        idx += 1
    finally:
      cap.release()
    if not frames:
      raise RuntimeError(f"No frames decoded from {video_path}")
    return frames

  def _latency_loop(
    self, model: YOLO | RTDETR, frames: list[np.ndarray]
  ) -> LatencyStats:
    device = self.config.device
    imgsz = self.config.image_size

    # Warm-up.
    for i in range(self.config.latency_warmup):
      model.predict(frames[i % len(frames)], imgsz=imgsz, device=device, verbose=False)
    if device.startswith("cuda"):
      torch.cuda.synchronize()

    samples_ms: list[float] = []
    for i in range(self.config.latency_iters):
      frame = frames[i % len(frames)]
      if device.startswith("cuda"):
        torch.cuda.synchronize()
      t0 = time.perf_counter()
      model.predict(frame, imgsz=imgsz, device=device, verbose=False)
      if device.startswith("cuda"):
        torch.cuda.synchronize()
      samples_ms.append((time.perf_counter() - t0) * 1000.0)
    return LatencyStats.from_samples(samples_ms)

  def _log_to_tracker(self, r: BenchmarkResult) -> None:
    flat = {
      "model": r.model_name,
      "device": r.device,
      "image_size": r.image_size,
      "params": r.info["parameters"],
      "gflops": r.info["gflops"] or 0.0,
      "weights_mb": r.info["weights_mb"],
      "map50": r.detection["map50"],
      "map50_95": r.detection["map50_95"],
      "precision": r.detection["precision"],
      "recall": r.detection["recall"],
      "f1": r.detection["f1"],
      "latency_mean_ms": r.latency["mean_ms"],
      "latency_p95_ms": r.latency["p95_ms"],
      "fps": r.latency["fps_mean"],
    }
    self.tracker.log_params({"model": r.model_name, "image_size": r.image_size})
    self.tracker.log_metrics(
      {k: v for k, v in flat.items() if isinstance(v, (int, float))}
    )

  def _write_json(self, r: BenchmarkResult) -> None:
    path = self.config.output_dir / f"{r.model_name}.json"
    path.write_text(json.dumps(r.to_dict(), indent=2))

  def _write_summary(self, results: list[BenchmarkResult]) -> None:
    if not results:
      return
    csv_path = self.config.output_dir / "summary.csv"
    rows = [
      {
        "model": r.model_name,
        "params": r.info["parameters"],
        "gflops": r.info["gflops"],
        "weights_mb": round(r.info["weights_mb"], 2),
        "map50": round(r.detection["map50"], 4),
        "map50_95": round(r.detection["map50_95"], 4),
        "precision": round(r.detection["precision"], 4),
        "recall": round(r.detection["recall"], 4),
        "f1": round(r.detection["f1"], 4),
        "latency_mean_ms": round(r.latency["mean_ms"], 3),
        "latency_p50_ms": round(r.latency["p50_ms"], 3),
        "latency_p95_ms": round(r.latency["p95_ms"], 3),
        "latency_p99_ms": round(r.latency["p99_ms"], 3),
        "fps": round(r.latency["fps_mean"], 2),
      }
      for r in results
    ]
    with open(csv_path, "w", newline="") as f:
      w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
      w.writeheader()
      w.writerows(rows)
    json_path = self.config.output_dir / "summary.json"
    json_path.write_text(json.dumps([r.to_dict() for r in results], indent=2))
    logger.success("Wrote benchmark summary: {} ({} models)", csv_path, len(results))
