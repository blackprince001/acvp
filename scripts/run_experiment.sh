#!/usr/bin/env bash
# End-to-end experiment driver.
#
# 1. Collect telemetry from each TRAIN_STREAM (heavy + medium/low traffic)
# 2. Collect telemetry from EVAL_STREAM (held out for benchmarking)
# 3. Train lstm + gru + tcn on the combined training CSVs
# 4. Evaluate each estimator offline on the eval CSV
#
# After this finishes, run scripts/inference/run_pipeline_live.sh against
# EVAL_STREAM with the best estimator for the end-to-end paper demo.
#
# Usage:
#   scripts/run_experiment.sh [detector] [duration_seconds]
#     detector         default: yolov8n
#     duration_seconds default: 14400 (4h) — applied per stream

set -euo pipefail

DETECTOR="${1:-yolov8n}"
DURATION="${2:-14400}"

# --- Streams ---------------------------------------------------------------
# Training pool: blend high-traffic Poway and lighter Seoul intersection so
# the estimator sees both regimes.
TRAIN_STREAMS=(
  "https://wzmedia.dot.ca.gov/D11/C043_SB_15_JSO_Poway_Rd.stream/playlist.m3u8"
  "https://topiscctv1.eseoul.go.kr/edge6/ch133.stream/playlist.m3u8"
)
# Held-out for evaluation + final live demo.
EVAL_STREAM="https://wzmedia.dot.ca.gov/D4/S101_at_Airport_Bl.stream/playlist.m3u8"
# ---------------------------------------------------------------------------

RUN_TAG="$(date +%Y%m%d_%H%M%S)_${DETECTOR}"
RUN_DIR="experiments/inference/live/${RUN_TAG}"
EST_ROOT="experiments/estimator/${RUN_TAG}"
mkdir -p "${RUN_DIR}" "${EST_ROOT}"

TRAIN_CSVS=()
for i in "${!TRAIN_STREAMS[@]}"; do
  csv="${RUN_DIR}/train_${i}.csv"
  TRAIN_CSVS+=("${csv}")
  echo "[1/4] Collect train stream ${i}"
  scripts/inference/collect_stream.sh "${TRAIN_STREAMS[$i]}" "${DURATION}" "${DETECTOR}" "${csv}"
done

EVAL_CSV="${RUN_DIR}/eval.csv"
echo "[2/4] Collect eval stream"
scripts/inference/collect_stream.sh "${EVAL_STREAM}" "${DURATION}" "${DETECTOR}" "${EVAL_CSV}"

echo "[3/4] Train estimators"
scripts/training/train_estimators.sh "${EST_ROOT}" "${TRAIN_CSVS[@]}"

echo "[4/4] Evaluate estimators on held-out stream"
scripts/evaluation/evaluate_estimators.sh "${EVAL_CSV}" "${EST_ROOT}"

echo
echo "Run summary: ${RUN_DIR}"
echo "Estimators : ${EST_ROOT}/{lstm,gru,tcn}/"
echo "Next       : scripts/inference/run_pipeline_live.sh ${DETECTOR} <best_est> ${EST_ROOT}/<best_est> 600"
