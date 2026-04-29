#!/usr/bin/env bash
# End-to-end live demo: detection -> tracking -> spatial filter -> features ->
# estimator forecast, all against the held-out EVAL_STREAM.
#
# Produces an annotated video and per-frame telemetry for paper figures.
#
# Usage:
#   scripts/inference/run_pipeline_live.sh <detector> <estimator> <estimator_dir> [duration]
#
#   detector       : yolov8n | yolov8s | yolov5l | ...
#   estimator      : lstm | gru | tcn
#   estimator_dir  : path containing best.pt + normalizer.npz
#   duration       : default 600 (10 minutes) — keep short for the demo
#
# Example:
#   scripts/inference/run_pipeline_live.sh yolov8n lstm \
#     experiments/estimator/lstm_yolov8n_20260429_120000_yolov8n 600

set -euo pipefail

DETECTOR="${1:?detector required}"
ESTIMATOR="${2:?estimator required}"
EST_DIR="${3:?estimator dir required}"
DURATION="${4:-600}"

# Keep in sync with scripts/run_experiment.sh.
EVAL_STREAM="https://wzmedia.dot.ca.gov/D4/S101_at_Airport_Bl.stream/playlist.m3u8"

WEIGHTS="experiments/detection/${DETECTOR}/weights/best.pt"
CKPT="${EST_DIR}/best.pt"
NORM="${EST_DIR}/normalizer.npz"
for f in "${WEIGHTS}" "${CKPT}" "${NORM}"; do
  [[ -f "${f}" ]] || { echo "Missing: ${f}" >&2; exit 1; }
done

TAG="$(date +%Y%m%d_%H%M%S)_${DETECTOR}_${ESTIMATOR}"
OUT_DIR="experiments/inference/pipeline_live/${TAG}"
mkdir -p "${OUT_DIR}"

uv run python scripts/inference/run_inference.py \
  --video "${EVAL_STREAM}" \
  --weights "${WEIGHTS}" \
  --mode predict \
  --estimator "${ESTIMATOR}" \
  --estimator-checkpoint "${CKPT}" \
  --normalizer-stats "${NORM}" \
  --input-len 10 --horizon 5 \
  --input-size 10 --output-size 10 \
  --duration "${DURATION}" \
  --output "${OUT_DIR}/annotated.mp4" \
  --telemetry "${OUT_DIR}/telemetry.csv" \
  --log-level INFO

echo "Done."
echo "  video    : ${OUT_DIR}/annotated.mp4"
echo "  telemetry: ${OUT_DIR}/telemetry.csv"
