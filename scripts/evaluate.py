#!/usr/bin/env python3
"""Evaluate detection model variants.

Discovers trained checkpoints under ``experiments/detection/<model>/weights/best.pt``,
runs the full benchmark suite (mAP, latency, model size) on each, then
generates paper figures from the resulting summary.

Examples
--------
Bench all trained models on the test split::

    uv run python scripts/evaluate.py

Bench a subset, smaller image size, log to TensorBoard::

    uv run python scripts/evaluate.py \\
        --models yolov8n yolov8s yolov11m \\
        --image-size 480 \\
        --tensorboard experiments/tb/bench-2026-04-26

Skip the latency loop (faster sanity run)::

    uv run python scripts/evaluate.py --latency-iters 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger  # noqa: E402

from src.evaluation.benchmark import BenchmarkConfig, DetectionBenchmark  # noqa: E402
from src.evaluation.visualization import render_all  # noqa: E402
from src.experiment.tracker import ExperimentTracker  # noqa: E402
from src.utils.logging import setup_logger  # noqa: E402

DEFAULT_DATA_YAML = PROJECT_ROOT / "data" / "yolo_detection" / "data.yaml"
DEFAULT_DETECTION_DIR = PROJECT_ROOT / "experiments" / "detection"


def discover_models(detection_dir: Path) -> dict[str, Path]:
  found: dict[str, Path] = {}
  for sub in sorted(detection_dir.iterdir()):
    if not sub.is_dir():
      continue
    weights = sub / "weights" / "best.pt"
    if weights.exists():
      found[sub.name] = weights
  return found


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Benchmark detection model variants.")
  p.add_argument("--data-yaml", type=Path, default=DEFAULT_DATA_YAML)
  p.add_argument("--detection-dir", type=Path, default=DEFAULT_DETECTION_DIR)
  p.add_argument("--output-dir", type=Path, default=None,
                 help="Output directory (default: experiments/benchmarks/<timestamp>).")
  p.add_argument("--models", nargs="+", default=None,
                 help="Subset of model dirs to bench (default: all with best.pt).")
  p.add_argument("--split", default="test", choices=["train", "val", "test"])
  p.add_argument("--image-size", type=int, default=640)
  p.add_argument("--device", default="cuda")
  p.add_argument("--batch", type=int, default=1)
  p.add_argument("--conf", type=float, default=0.001)
  p.add_argument("--iou", type=float, default=0.6)
  p.add_argument("--latency-warmup", type=int, default=10)
  p.add_argument("--latency-iters", type=int, default=100)
  p.add_argument("--latency-max-images", type=int, default=200,
                 help="Cap on frames decoded for the latency loop (image or video source).")
  p.add_argument("--latency-video", type=Path, default=None,
                 help="If set, latency loop reads frames from this video instead of "
                      "the test split (mAP still uses the test split).")
  p.add_argument("--latency-video-stride", type=int, default=1,
                 help="Decode every Nth frame from --latency-video (default 1).")
  p.add_argument("--tensorboard", type=Path, default=None,
                 help="TensorBoard log dir. If unset, no TB logging.")
  p.add_argument("--wandb-project", default=None)
  p.add_argument("--wandb-run", default=None)
  p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
  p.add_argument("--no-figures", action="store_true",
                 help="Skip figure rendering after the bench.")
  p.add_argument("--log-level", default="INFO")
  return p.parse_args()


def build_tracker(args: argparse.Namespace) -> ExperimentTracker:
  cfg: dict = {}
  if args.tensorboard:
    cfg["tensorboard"] = {"log_dir": str(args.tensorboard)}
  if args.wandb_project:
    cfg["wandb"] = {
      "project": args.wandb_project,
      "run_name": args.wandb_run,
      "mode": args.wandb_mode,
    }
  return ExperimentTracker.from_config(cfg)


def main() -> int:
  args = parse_args()
  setup_logger(log_level=args.log_level, log_file="experiments/logs/evaluate.log")

  if args.output_dir is None:
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.output_dir = PROJECT_ROOT / "experiments" / "benchmarks" / stamp
  args.output_dir.mkdir(parents=True, exist_ok=True)

  available = discover_models(args.detection_dir)
  if not available:
    logger.error("No trained models with best.pt under {}", args.detection_dir)
    return 1
  if args.models:
    missing = [m for m in args.models if m not in available]
    for m in missing:
      logger.warning("Requested model {} not found — skipping", m)
    targets = {m: available[m] for m in args.models if m in available}
  else:
    targets = available
  if not targets:
    logger.error("No models to benchmark.")
    return 1

  logger.info("Benchmarking {} model(s): {}", len(targets), list(targets))

  cfg = BenchmarkConfig(
    data_yaml=args.data_yaml,
    output_dir=args.output_dir,
    split=args.split,
    image_size=args.image_size,
    device=args.device,
    batch=args.batch,
    conf=args.conf,
    iou=args.iou,
    latency_warmup=args.latency_warmup,
    latency_iters=args.latency_iters,
    latency_max_images=args.latency_max_images,
    latency_video=args.latency_video,
    latency_video_stride=args.latency_video_stride,
  )
  tracker = build_tracker(args)
  tracker.log_params(
    {
      "split": args.split,
      "image_size": args.image_size,
      "device": args.device,
      "n_models": len(targets),
    }
  )

  bench = DetectionBenchmark(cfg, tracker=tracker)
  results = bench.run_many(targets)

  if not args.no_figures and results:
    figs_dir = args.output_dir / "figures"
    figs = render_all(args.output_dir / "summary.json", figs_dir)
    for name, path in figs.items():
      logger.info("Figure {} → {}", name, path)
      tracker.log_artifact(path, name=f"fig_{name}")

  tracker.close()
  logger.success("Done. Results in {}", args.output_dir)
  return 0


if __name__ == "__main__":
  sys.exit(main())
