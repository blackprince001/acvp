#!/usr/bin/env bash
# Record per-frame telemetry from a single live stream.
#
# Usage:
#   scripts/inference/collect_stream.sh <stream_url> <duration_seconds> <detector> <out_csv>
#
# Example:
#   scripts/inference/collect_stream.sh \
#     "https://wzmedia.dot.ca.gov/D11/C043_SB_15_JSO_Poway_Rd.stream/playlist.m3u8" \
#     14400 yolov8n experiments/inference/live/poway/telemetry.csv

set -euo pipefail

URL="${1:?stream url required}"
DURATION="${2:?duration seconds required}"
DETECTOR="${3:?detector required}"
OUT_CSV="${4:?output csv path required}"

WEIGHTS="experiments/detection/${DETECTOR}/weights/best.pt"
if [[ ! -f "${WEIGHTS}" ]]; then
  echo "Weights not found: ${WEIGHTS}" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUT_CSV}")"

echo "Collecting ${DURATION}s of telemetry"
echo "  detector : ${DETECTOR}"
echo "  stream   : ${URL}"
echo "  csv      : ${OUT_CSV}"

uv run python scripts/inference/run_inference.py \
  --video "${URL}" \
  --weights "${WEIGHTS}" \
  --duration "${DURATION}" \
  --telemetry "${OUT_CSV}" \
  --log-level INFO
