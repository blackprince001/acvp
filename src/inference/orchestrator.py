"""Inference pipeline orchestrator.

Coordinates detection, tracking, spatial filtering, feature extraction, and
optional density prediction on a single video source.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

from src.features.extractor import FEATURE_NAMES, FeatureExtractor
from src.features.normalizer import FeatureNormalizer
from src.features.windowing import WindowBuilder
from src.models.ml_estimator.base import BaseEstimator
from src.models.ml_estimator.gru import GRUEstimator
from src.models.ml_estimator.lstm import LSTMEstimator
from src.models.ml_estimator.tcn import TCNEstimator
from src.spatial_logic.filtering import OnRoadFilter
from src.spatial_logic.road_mask import RoadMaskManager
from src.tracking.ultralytics_tracker import UltralyticsTracker
from src.utils.types import FilteredResult, PredictionResult, TrackingResult

from .fps_controller import FixedFPSController
from .video_io import VideoReader, VideoWriter, draw_overlay

_ESTIMATOR_REGISTRY: dict[str, type[BaseEstimator]] = {
  "lstm": LSTMEstimator,
  "gru": GRUEstimator,
  "tcn": TCNEstimator,
}

_VALID_MODES = {"detect", "segment", "predict"}


@dataclass
class FrameOutput:
  """Per-frame result bundle emitted by the orchestrator."""

  frame_idx: int
  timestamp: float
  tracking: TrackingResult | None
  filtered: FilteredResult | None
  features: np.ndarray | None
  prediction: PredictionResult | None


class InferenceOrchestrator:
  """Run the full CV + (optional) ML estimator pipeline over a video source.

  Config schema (relevant keys, all optional unless noted):

      mode: "detect" | "segment" | "predict"  (default "detect")
      device: "cuda" | "cpu" | "mps"          (default "cuda" if available)
      target_fps: float | null                 (null disables FPS pacing)

      detection:
        weights: path to YOLO .pt used for tracking + detection
        tracker: "botsort" | "bytetrack"       (default "botsort")
        track_buffer: int                       (default 30)
        match_thresh: float                     (default 0.8)
        track_low_thresh: float                 (default 0.1)

      road_mask:
        static_path: path to .npy or image      (optional)

      filtering:
        overlap_threshold: float                (default 0.5)
        min_speed: float                        (default 0.0)

      features:
        max_speed: float                        (default 50.0)
        stop_threshold: float                   (default 1.0)

      estimator:  # required when mode == "predict"
        name: "lstm" | "gru" | "tcn"
        checkpoint: path to .pt
        input_size, output_size, horizon: ints
        input_len: int                          (default 10)
        normalizer_stats: optional path to .npz with "center" and "scale"

      output:
        video_path: output overlay video        (optional)
        telemetry_path: CSV path                (optional)
  """

  def __init__(self, config: dict[str, Any]) -> None:
    self.config = config
    self.mode: str = config.get("mode", "detect")
    if self.mode not in _VALID_MODES:
      raise ValueError(f"mode must be one of {_VALID_MODES}, got {self.mode!r}")

    self.device: str = config.get(
      "device", "cuda" if torch.cuda.is_available() else "cpu"
    )

    self.target_fps: float | None = config.get("target_fps")

    # Detection + tracking: Ultralytics handles both in one call.
    det_cfg = config.get("detection", {}) or {}
    tracker_cfg = {
      "tracker_type": det_cfg.get("tracker", "botsort"),
      "track_buffer": det_cfg.get("track_buffer", 30),
      "match_thresh": det_cfg.get("match_thresh", 0.8),
      "track_low_thresh": det_cfg.get("track_low_thresh", 0.1),
      "track_high_thresh": det_cfg.get("track_high_thresh", 0.5),
      "new_track_thresh": det_cfg.get("new_track_thresh", 0.6),
      "fuse_score": det_cfg.get("fuse_score", True),
    }
    weights = det_cfg.get("weights", "yolov8n.pt")
    self.tracker = UltralyticsTracker(
      config=tracker_cfg, model_path=weights, device=self.device
    )

    # Road mask manager
    self.mask_manager = RoadMaskManager()
    mask_path = (config.get("road_mask") or {}).get("static_path")
    if mask_path:
      self.mask_manager.load_static(mask_path)
      logger.info("Loaded static road mask from {}", mask_path)

    # On-road filter
    filt_cfg = config.get("filtering", {}) or {}
    self.on_road_filter = OnRoadFilter(
      overlap_threshold=filt_cfg.get("overlap_threshold", 0.5),
      min_speed=filt_cfg.get("min_speed", 0.0),
    )

    # Feature extractor
    feat_cfg = config.get("features", {}) or {}
    self.feature_extractor = FeatureExtractor(
      max_speed=feat_cfg.get("max_speed", 50.0),
      stop_threshold=feat_cfg.get("stop_threshold", 1.0),
    )

    # Estimator (predict mode only)
    self.estimator: BaseEstimator | None = None
    self.window_builder: WindowBuilder | None = None
    self.normalizer: FeatureNormalizer | None = None
    self.input_len: int = 0
    self.horizon: int = 0

    if self.mode == "predict":
      self._load_estimator(config.get("estimator") or {})

    self.fps_controller = FixedFPSController(target_fps=self.target_fps)

    logger.info(
      "InferenceOrchestrator: mode={} device={} target_fps={}",
      self.mode,
      self.device,
      self.target_fps,
    )

  def run(
    self,
    video_source: str | Path,
    output_path: str | Path | None = None,
    telemetry_path: str | Path | None = None,
    max_frames: int | None = None,
  ) -> list[FrameOutput]:
    """Run the pipeline on *video_source* and return per-frame outputs.

    Args:
        video_source: Path to a video file or directory of images.
        output_path: Optional path for the annotated output video.
        telemetry_path: Optional CSV path to log per-frame features.
        max_frames: Stop after this many frames (useful for smoke tests).
    """
    output_path = output_path or (self.config.get("output") or {}).get("video_path")
    telemetry_path = telemetry_path or (self.config.get("output") or {}).get(
      "telemetry_path"
    )

    outputs: list[FrameOutput] = []
    feature_buffer: list[np.ndarray] = []
    writer: VideoWriter | None = None
    telemetry_file = None
    telemetry_csv: csv.writer | None = None

    reader = VideoReader(video_source)
    try:
      if output_path is not None:
        writer = VideoWriter(
          output_path, width=reader.width, height=reader.height, fps=reader.fps
        )
      if telemetry_path is not None:
        telemetry_file = open(telemetry_path, "w", newline="")
        telemetry_csv = csv.writer(telemetry_file)
        telemetry_csv.writerow(
          ["frame_idx", "timestamp", "vehicle_count", "occupancy_ratio", *FEATURE_NAMES]
        )

      self.fps_controller.start()

      for frame_idx, timestamp, frame in reader.frames():
        if max_frames is not None and frame_idx >= max_frames:
          break
        if self.fps_controller.should_skip():
          continue

        out = self._process_frame(frame_idx, timestamp, frame, feature_buffer)
        outputs.append(out)

        if writer is not None:
          road_mask = self._try_get_mask(frame.shape[:2], frame_idx)
          annotated = draw_overlay(
            frame,
            tracking=out.tracking,
            filtered=out.filtered,
            prediction=out.prediction,
            road_mask=road_mask,
          )
          writer.write(annotated)

        if (
          telemetry_csv is not None
          and out.features is not None
          and out.filtered is not None
        ):
          telemetry_csv.writerow(
            [
              out.frame_idx,
              out.timestamp,
              out.filtered.vehicle_count,
              out.filtered.occupancy_ratio,
              *[float(v) for v in out.features],
            ]
          )

        self.fps_controller.tick()

      self.fps_controller.log_summary()
    finally:
      reader.close()
      if writer is not None:
        writer.close()
      if telemetry_file is not None:
        telemetry_file.close()

    return outputs

  def _process_frame(
    self,
    frame_idx: int,
    timestamp: float,
    frame: np.ndarray,
    feature_buffer: list[np.ndarray],
  ) -> FrameOutput:
    # 1. Detect + track (Ultralytics runs detection internally).
    tracking_out = self.tracker.update(
      detections=np.empty((0, 4)),
      confidences=np.empty((0,)),
      class_ids=np.empty((0,), dtype=int),
      frame=frame,
    )
    tracking_result = TrackingResult(
      frame_idx=frame_idx,
      timestamp=timestamp,
      tracked_objects=[
        {
          "track_id": t.track_id,
          "box": t.bbox,
          "centroid": t.centroid,
          "confidence": t.confidence,
          "class_id": t.class_id,
          "speed": t.speed,
          "direction": t.direction,
        }
        for t in tracking_out.tracks
      ],
    )

    # 2. Spatial filtering needs a road mask. Fall back to full-frame if none.
    road_mask = self._try_get_mask(frame.shape[:2], frame_idx)
    if road_mask is None:
      road_mask = np.ones(frame.shape[:2], dtype=bool)
    filtered = self.on_road_filter.filter(
      tracking_out, road_mask, frame_idx=frame_idx, timestamp=timestamp
    )

    # 3. Feature extraction (flow = 0 placeholder unless a counter is wired in).
    features = self.feature_extractor.extract(filtered, road_mask, flow=0.0)
    feature_buffer.append(features)

    # 4. Prediction (if enabled and enough history).
    prediction = None
    if self.mode == "predict" and self.estimator is not None:
      prediction = self._maybe_predict(feature_buffer, frame_idx, timestamp)

    return FrameOutput(
      frame_idx=frame_idx,
      timestamp=timestamp,
      tracking=tracking_result,
      filtered=filtered,
      features=features,
      prediction=prediction,
    )

  def _maybe_predict(
    self,
    feature_buffer: list[np.ndarray],
    frame_idx: int,
    timestamp: float,
  ) -> PredictionResult | None:
    if len(feature_buffer) < self.input_len:
      return None
    assert self.estimator is not None

    window = np.stack(feature_buffer[-self.input_len :])  # (T, F)
    if self.normalizer is not None:
      window_input = self.normalizer.transform(window)
    else:
      window_input = window

    preds = self.estimator.predict(window_input)[0]  # (horizon, F_out)
    if self.normalizer is not None and preds.shape[-1] == window.shape[-1]:
      preds = self.normalizer.inverse_transform(preds)

    return PredictionResult(
      window_end=frame_idx,
      timestamp=timestamp,
      current_density=float(window[-1, 0]),  # vehicle_count as stand-in density
      predicted_density=preds,
      confidence=np.ones(preds.shape[0], dtype=float),
    )

  def _load_estimator(self, est_cfg: dict[str, Any]) -> None:
    name = est_cfg.get("name")
    if name not in _ESTIMATOR_REGISTRY:
      raise ValueError(
        f"estimator.name must be one of {sorted(_ESTIMATOR_REGISTRY)}, got {name!r}"
      )
    cls = _ESTIMATOR_REGISTRY[name]
    model = cls(est_cfg)
    ckpt = est_cfg.get("checkpoint")
    if ckpt:
      state = torch.load(ckpt, map_location=self.device)
      model.load_state_dict(state.get("model_state_dict", state))
      logger.info("Loaded {} estimator from {}", name, ckpt)
    model.to(self.device)
    model.eval()

    self.estimator = model
    self.input_len = est_cfg.get("input_len", 10)
    self.horizon = est_cfg["horizon"]
    self.window_builder = WindowBuilder(input_len=self.input_len, horizon=self.horizon)

    stats_path = est_cfg.get("normalizer_stats")
    if stats_path:
      stats = np.load(stats_path)
      norm = FeatureNormalizer(method=est_cfg.get("normalizer_method", "standard"))
      norm._center = stats["center"]
      norm._scale = stats["scale"]
      norm._fitted = True
      self.normalizer = norm
      logger.info("Loaded normalizer stats from {}", stats_path)

  def _try_get_mask(self, shape: tuple[int, int], frame_idx: int) -> np.ndarray | None:
    try:
      mask = self.mask_manager.get_mask(frame_idx)
    except RuntimeError:
      return None
    if mask.shape[:2] != shape:
      # Silent resize is a pipeline concern; skip to avoid accidental mismatches.
      logger.warning(
        "Road mask shape {} does not match frame shape {}; ignoring for this frame.",
        mask.shape[:2],
        shape,
      )
      return None
    return mask
