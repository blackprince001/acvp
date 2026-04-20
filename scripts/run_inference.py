#!/usr/bin/env python3
"""
Inference entry point — Phase 7.

Runs the detection + tracking + spatial filtering + feature extraction
pipeline on a video source, optionally invoking the ML density
estimator. Writes an overlay video and per-frame telemetry CSV.

Usage examples
--------------
Detection-only (no prediction)::

    uv run python scripts/05_run_inference.py \\
        --video data/raw/scene_001/video.mp4 \\
        --weights experiments/detection/yolov8n/weights/best.pt \\
        --output experiments/inference/scene_001.mp4 \\
        --telemetry experiments/inference/scene_001.csv

With density prediction::

    uv run python scripts/05_run_inference.py \\
        --video data/raw/scene_001/video.mp4 \\
        --weights experiments/detection/yolov8n/weights/best.pt \\
        --mode predict \\
        --estimator lstm \\
        --estimator-checkpoint experiments/estimator/lstm/best.pt \\
        --input-len 10 --horizon 5 \\
        --input-size 10 --output-size 10

Quick smoke test (30 frames, no output)::

    uv run python scripts/05_run_inference.py \\
        --video data/raw/scene_001/video.mp4 \\
        --weights yolov8n.pt --max-frames 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger  # noqa: E402

from src.inference.orchestrator import InferenceOrchestrator  # noqa: E402
from src.utils.logging import setup_logger  # noqa: E402


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Run the traffic density inference pipeline.")
  p.add_argument(
    "--video", required=True, help="Path to video file or image directory."
  )
  p.add_argument(
    "--weights", required=True, help="YOLO detection+tracking weights (.pt)."
  )
  p.add_argument(
    "--mode",
    choices=["detect", "segment", "predict"],
    default="detect",
    help="Pipeline mode.",
  )
  p.add_argument("--device", default=None, help="Compute device (cuda/cpu/mps).")
  p.add_argument("--target-fps", type=float, default=None, help="Target output FPS.")
  p.add_argument("--max-frames", type=int, default=None, help="Stop after N frames.")

  p.add_argument("--tracker", default="botsort", choices=["botsort", "bytetrack"])
  p.add_argument("--road-mask", default=None, help="Static road mask (.npy or image).")
  p.add_argument("--overlap-threshold", type=float, default=0.5)
  p.add_argument("--min-speed", type=float, default=0.0)

  # Estimator options (predict mode)
  p.add_argument("--estimator", choices=["lstm", "gru", "tcn"], default=None)
  p.add_argument("--estimator-checkpoint", default=None)
  p.add_argument("--input-len", type=int, default=10)
  p.add_argument("--horizon", type=int, default=5)
  p.add_argument("--input-size", type=int, default=10)
  p.add_argument("--output-size", type=int, default=10)
  p.add_argument("--normalizer-stats", default=None)

  # Output options
  p.add_argument("--output", default=None, help="Annotated output video path.")
  p.add_argument("--telemetry", default=None, help="Telemetry CSV output path.")

  p.add_argument("--log-level", default="INFO")
  return p.parse_args()


def build_config(args: argparse.Namespace) -> dict:
  cfg: dict = {
    "mode": args.mode,
    "target_fps": args.target_fps,
    "detection": {
      "weights": args.weights,
      "tracker": args.tracker,
    },
    "filtering": {
      "overlap_threshold": args.overlap_threshold,
      "min_speed": args.min_speed,
    },
    "output": {
      "video_path": args.output,
      "telemetry_path": args.telemetry,
    },
  }
  if args.device:
    cfg["device"] = args.device
  if args.road_mask:
    cfg["road_mask"] = {"static_path": args.road_mask}
  if args.mode == "predict":
    if not args.estimator or not args.estimator_checkpoint:
      raise SystemExit(
        "--estimator and --estimator-checkpoint are required in predict mode"
      )
    cfg["estimator"] = {
      "name": args.estimator,
      "checkpoint": args.estimator_checkpoint,
      "input_size": args.input_size,
      "output_size": args.output_size,
      "horizon": args.horizon,
      "input_len": args.input_len,
      "normalizer_stats": args.normalizer_stats,
    }
  return cfg


def main() -> int:
  args = parse_args()
  setup_logger(log_level=args.log_level, log_file="experiments/logs/inference.log")

  cfg = build_config(args)
  logger.info("Config: {}", cfg)

  orchestrator = InferenceOrchestrator(cfg)
  outputs = orchestrator.run(
    video_source=args.video,
    output_path=args.output,
    telemetry_path=args.telemetry,
    max_frames=args.max_frames,
  )
  logger.info("Processed {} frames", len(outputs))
  return 0


if __name__ == "__main__":
  sys.exit(main())
