#!/usr/bin/env bash
# Train lstm + gru + tcn on one or more telemetry CSVs.
# Each estimator is windowed per-CSV (no cross-stream splicing).
#
# Usage:
#   scripts/training/train_estimators.sh <output_root> <train_csv_1> [train_csv_2 ...]
#
# Example:
#   scripts/training/train_estimators.sh experiments/estimator/run01 \
#     experiments/inference/live/poway/telemetry.csv \
#     experiments/inference/live/seoul/telemetry.csv

set -euo pipefail

OUTPUT_ROOT="${1:?output root required}"
shift
if [[ "$#" -lt 1 ]]; then
  echo "At least one training CSV is required." >&2
  exit 1
fi
CSVS=("$@")

mkdir -p "${OUTPUT_ROOT}"

for EST in lstm gru tcn; do
  EST_DIR="${OUTPUT_ROOT}/${EST}"
  mkdir -p "${EST_DIR}"
  echo "Training ${EST} on ${#CSVS[@]} CSV(s) -> ${EST_DIR}"
  uv run python scripts/training/train_estimator.py \
    --csv "${CSVS[@]}" \
    --model "${EST}" \
    --output-dir "${EST_DIR}" \
    --input-len 10 \
    --horizon 5 \
    --epochs 200 \
    --patience 30
done

echo "Done. Estimators in ${OUTPUT_ROOT}/{lstm,gru,tcn}/"
