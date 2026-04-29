#!/usr/bin/env bash
# Evaluate lstm + gru + tcn checkpoints against a held-out telemetry CSV.
# Expects each estimator dir to contain best.pt + normalizer.npz, as produced
# by scripts/training/train_estimators.sh.
#
# Usage:
#   scripts/evaluation/evaluate_estimators.sh <eval_csv> <estimators_root>
#
# Example:
#   scripts/evaluation/evaluate_estimators.sh \
#     experiments/inference/live/airport/eval.csv \
#     experiments/estimator/run01

set -euo pipefail

EVAL_CSV="${1:?eval csv required}"
ROOT="${2:?estimators root required}"

for EST in lstm gru tcn; do
  EST_DIR="${ROOT}/${EST}"
  if [[ ! -f "${EST_DIR}/best.pt" || ! -f "${EST_DIR}/normalizer.npz" ]]; then
    echo "Skipping ${EST}: ${EST_DIR}/best.pt or normalizer.npz missing." >&2
    continue
  fi
  echo "Evaluating ${EST} on ${EVAL_CSV}"
  uv run python scripts/evaluation/evaluate_estimator.py \
    --csv "${EVAL_CSV}" \
    --model "${EST}" \
    --checkpoint "${EST_DIR}/best.pt" \
    --normalizer "${EST_DIR}/normalizer.npz" \
    --input-len 10 \
    --horizon 5
done

echo "Done. metrics.json + predictions.npz written to each estimator dir."
