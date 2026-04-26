# Feature: Traffic Level Classifier (LOW / MEDIUM / HIGH)

**Status:** planned, not implemented yet.
**Trigger:** runs as a tail-stage of `scripts/run_inference.py` once the
detection + tracking + estimator pipeline is producing telemetry.
**Owner:** TBD.

## Goal

Translate the orchestrator's per-frame feature vector and (optional)
estimator forecast into a compact categorical level — `LOW`, `MEDIUM`,
or `HIGH` — that can drive a live dashboard without consumers needing to
understand the underlying 10-feature vector.

## Inputs

Per frame, available from `InferenceOrchestrator`:

- `FilteredResult.vehicle_count`, `occupancy_ratio`
- `features` ndarray (10 floats, see `src/features/extractor.py`)
- `PredictionResult.predicted_density` of shape `(horizon, output_size)`
  when the estimator is wired in

## Outputs

For each frame:

```json
{
  "frame_idx": 1234,
  "timestamp": 41.13,
  "level": "MEDIUM",
  "confidence": 0.62,
  "drivers": {"congestion_index": 0.31, "occupancy_ratio": 0.27},
  "forecast_levels": ["MEDIUM", "MEDIUM", "HIGH", "HIGH", "HIGH"]
}
```

Streamed to:
1. The video overlay (colored chip top-right).
2. A JSONL file alongside the existing telemetry CSV (`*.levels.jsonl`).
3. Optionally, a WebSocket later — JSONL is the v1 contract.

## Classification strategy

Two modes, picked by config.

### 1. Threshold mode (default, no training required)

Single signal: `congestion_index` (already in `[0, 1]`). Defaults:

| Range            | Level   |
|------------------|---------|
| `ci < 0.15`      | LOW     |
| `0.15 ≤ ci < 0.45` | MEDIUM  |
| `ci ≥ 0.45`      | HIGH    |

Optional fallback when `vehicle_count == 0`: force LOW regardless of
other signals (avoids noise on empty roads).

### 2. Per-scene quantile mode

Once a few minutes of telemetry exist for a given camera:
- Compute 33rd / 66th percentile of `congestion_index` over a rolling
  window (e.g. last 30 minutes) or from a one-time calibration pass.
- Use those as cut points. Re-fit periodically.

This auto-adapts to scene type (highway vs urban) without manual tuning.

## Non-goals (explicitly out of scope for v1)

- ML-based level classification — overkill for a 3-bucket label.
- Per-lane breakdown — pipeline is not lane-aware.
- Smoothing / hysteresis — first cut is per-frame; viewers can smooth.
- Prediction confidence intervals — `confidence` is a stand-in derived
  from how close `congestion_index` is to a bucket boundary.

## Implementation sketch

```
src/features/traffic_level.py
  class TrafficLevelClassifier:
    def __init__(self, mode: str = "threshold", thresholds=(0.15, 0.45)): ...
    def classify(self, features: np.ndarray) -> tuple[str, float]: ...
    def classify_forecast(self, predicted: np.ndarray) -> list[str]: ...

src/inference/orchestrator.py
  - construct classifier from config["traffic_level"]
  - call per frame, attach result to FrameOutput
  - write to telemetry JSONL

src/inference/video_io.draw_overlay
  - render colored chip (green/amber/red) from current level
```

## Acceptance criteria

- `TrafficLevelClassifier(mode="threshold").classify(...)` returns a
  valid label for any 10-feature vector (incl. all-zero).
- Running inference with `--traffic-level threshold` produces a
  `*.levels.jsonl` file with one record per processed frame.
- Overlay chip is visible top-right and switches color correctly on a
  hand-crafted synthetic clip (verified in tests).
- Quantile mode can be calibrated from an existing telemetry CSV via a
  small helper (`TrafficLevelClassifier.calibrate_from_csv(path)`).

## Dashboard hand-off

The dashboard team consumes the JSONL feed only. Treat it as a stable
contract: don't reorder keys, don't drop the `drivers` block, and
version-bump the schema if it changes.
