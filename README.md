# Real-Time Traffic Density Estimation

A vision pipeline that detects and tracks vehicles in a video stream, derives
per-frame traffic features (vehicle count, occupancy, density, congestion,
speed/direction statistics), and forecasts those features a few seconds ahead
with a sequence model (LSTM / GRU / TCN).

The system is designed around live HLS streams (e.g. Caltrans D4/D11 traffic
cameras) but works on local files and image directories too.

## Pipeline

```text
HLS / file ──▶ YOLO detect+track ──▶ on-road filter ──▶ feature extractor
                                                                │
                                                                ▼
                                                  ┌───────── per-frame CSV
                                                  │
                                                  ▼
                                       LSTM / GRU / TCN forecast
                                                  │
                                                  ▼
                                          annotated overlay + CSV
```

## Layout

| Path | What's there |
| --- | --- |
| `src/inference/` | Orchestrator, video I/O (URL-aware), FPS controller |
| `src/tracking/` | Ultralytics-backed detect+track |
| `src/spatial_logic/` | Road mask + on-road filtering |
| `src/features/` | Feature extractor, normalizer, sliding-window builder |
| `src/models/ml_estimator/` | LSTM / GRU / TCN density estimators |
| `scripts/data/` | Data preparation (Roboflow segmentation) |
| `scripts/training/` | Detector and estimator training |
| `scripts/inference/` | Live collection, predict-mode pipeline |
| `scripts/evaluation/` | Detection benchmark, estimator metrics, plots |
| `scripts/run_experiment.sh` | End-to-end orchestrator |
| `configs/` | YAML configs for detection + estimator hyperparams |
| `experiments/detection/<model>/` | Trained YOLO weights + training metrics |
| `experiments/estimator/<run>/<model>/` | Trained estimator + normalizer + eval metrics |

## Setup

Requires Python 3.12+ and a CUDA-capable GPU is recommended (CPU works but is slow).

```bash
# uv handles the virtualenv + lockfile resolution.
uv sync
```

## Training the detection model

The live pipeline depends on a YOLO `.pt` weight file that recognises vehicles
(and optionally road regions). If you don't have one, this is the upstream
flow:

### 1. Drop raw clips into `data/raw/`

Any `.mp4` works. The repo has a few sample clips committed under that path.

### 2. Annotate via Roboflow

`src/data-processing/roboflow_segmentation.py` wraps the Roboflow API to send
sampled frames to a hosted segmentation model and pulls back masks +
COCO-format annotations. Configure your workspace / project / API key in
`configs/base.yaml` (or pass via env), then run:

```bash
uv run python scripts/data/run_segmentation.py
```

Output lands in `data/segmented/` as `annotations.json` (COCO) plus per-frame
images. Two classes are expected: `road` (id 1) and `vehicles` (id 2).

If you'd rather hand-label, any COCO-format `data/segmented/annotations.json`
with the same class IDs works — Roboflow is just a convenience.

### 3. Convert to YOLO format and train

```bash
# Train one detector
uv run python scripts/training/train_detection.py --model yolov8n

# Sweep several variants (each writes to experiments/detection/<model>/)
uv run python scripts/training/train_detection.py --model yolov8n yolov8s yolov5l

# Common overrides
uv run python scripts/training/train_detection.py \
  --model yolov8n --epochs 50 --batch 16 --image-size 640 --device cuda
```

The script (a) re-indexes the COCO annotations to 0-based YOLO classes
(`road=0, vehicles=1`), (b) symlinks images into `data/yolo_detection/`,
(c) writes `data.yaml`, and (d) hands off to Ultralytics for training.

Per-model artifacts end up in `experiments/detection/<model>/`:

- `weights/best.pt`, `weights/last.pt` — checkpoints
- `results.csv` — epoch-by-epoch metrics (mAP, precision, recall, losses)
- PR / F1 / confusion-matrix plots written by Ultralytics

The `weights/best.pt` produced here is what `scripts/inference/run_inference.py`
expects via `--weights` (or what `collect_stream.sh <detector>` resolves
through `experiments/detection/<detector>/weights/best.pt`).

Supported model names: any `yolov5{n,s,m,l,x}`, `yolov8{n,s,m,l,x}`,
`yolov9{s,m,c,e}`, `yolov10{n,s,m,l,x,b}`, `yolo11{n,s,m,l,x}`,
`rtdetr-{l,x}` — see `ALL_MODELS` in `scripts/training/train_detection.py`.

## End-to-end workflow

The full pipeline is broken into composable shell scripts. Each can be run
standalone; `run_experiment.sh` chains them.

### 1. Collect telemetry from one stream

```bash
scripts/inference/collect_stream.sh \
  "<HLS_URL>" \
  <duration_seconds> \
  <detector_name> \
  <output_csv>
```

Example:

```bash
scripts/inference/collect_stream.sh \
  "https://wzmedia.dot.ca.gov/D11/C043_SB_15_JSO_Poway_Rd.stream/playlist.m3u8" \
  14400 yolov8n \
  experiments/inference/live/poway.csv
```

`<detector_name>` is a folder under `experiments/detection/` containing
`weights/best.pt` (e.g. `yolov8n`, `yolov8s`, `yolov5l`).

### 2. Train the three estimators

```bash
scripts/training/train_estimators.sh <output_root> <train_csv_1> [<train_csv_2> ...]
```

Each CSV is windowed independently — no cross-stream splicing — then concatenated.
Output: `<output_root>/{lstm,gru,tcn}/{best.pt,normalizer.npz}`.

### 3. Evaluate against a held-out CSV

```bash
scripts/evaluation/evaluate_estimators.sh <eval_csv> <output_root>
```

Writes `metrics.json` + `predictions.npz` into each estimator subdirectory.
Metrics include overall and per-horizon-step MSE/MAE/RMSE/R² in original
feature units.

### 4. Plot for the paper

```bash
uv run python scripts/evaluation/plot_estimator_results.py \
  --root <output_root> \
  --output-dir <output_root>/figures
```

Produces:

- `predicted_vs_actual_<feature>.png` — time-series, three models overlaid
- `scatter_<feature>.png` — prediction-vs-truth scatter (3 panels)
- `rmse_per_horizon.png` — RMSE as a function of forecast step
- `rmse_per_feature.png` — per-feature RMSE bars (log scale)
- `metrics_summary.csv` — flat table for the paper

### 5. Top-to-bottom live demo (annotated video)

```bash
scripts/inference/run_pipeline_live.sh <detector> <estimator> <estimator_dir> [duration]
```

Runs detection + tracking + estimator forecast against the held-out stream,
writing `annotated.mp4` and a per-frame `telemetry.csv` (with prediction
columns appended).

### Or do everything at once

```bash
scripts/run_experiment.sh [detector] [duration_seconds]
```

The orchestrator records both training streams + the eval stream, trains
all three estimators, and evaluates them. Edit the `TRAIN_STREAMS` /
`EVAL_STREAM` arrays at the top of the script to point at your own cameras.

## Standalone Python entry points

```bash
# Inference (detection-only or predict mode)
uv run python scripts/inference/run_inference.py \
  --video <file_or_url> \
  --weights experiments/detection/yolov8n/weights/best.pt \
  --duration 600 \
  --telemetry out.csv

# Train one estimator on N CSVs
uv run python scripts/training/train_estimator.py \
  --csv a.csv b.csv \
  --model lstm \
  --output-dir experiments/estimator/run01/lstm

# Evaluate one estimator
uv run python scripts/evaluation/evaluate_estimator.py \
  --csv eval.csv \
  --model lstm \
  --checkpoint experiments/estimator/run01/lstm/best.pt \
  --normalizer experiments/estimator/run01/lstm/normalizer.npz
```

`run_inference.py` accepts:

- `--video <path|url>` — local file, image directory, or HLS/RTSP URL
- `--duration <seconds>` — wall-clock cap (use this for live streams)
- `--max-frames <N>` — frame-count cap
- `--mode {detect,segment,predict}` — `predict` enables the estimator
- `--estimator {lstm,gru,tcn}`, `--estimator-checkpoint`, `--normalizer-stats`

## Bringing your own data

To run with a custom video or stream, you only need:

1. **A YOLO `.pt` weight file** for vehicle detection. Drop it under
   `experiments/detection/<name>/weights/best.pt` and pass `<name>` as the
   detector argument. (Or pass `--weights` directly.)
2. **A video source.** Local file, directory of frames, or a public HLS URL.
3. *(Optional)* **A road mask** (`.npy` boolean or grayscale image, same
   resolution as the frames) to restrict density/occupancy to the drivable
   area. Pass via `--road-mask`. Without one, the pipeline falls back to a
   full-frame mask and logs a warning.

Then collect → train → evaluate using the scripts above.

## Output schema

### Telemetry CSV (detect mode)

```text
frame_idx, timestamp, vehicle_count, occupancy_ratio,
<10 feature columns: vehicle_count, occupancy_ratio, mean_speed,
 mean_direction, density, flow, congestion_index,
 stopped_vehicle_ratio, speed_variance, direction_variance>
```

### Telemetry CSV (predict mode)

Same as above plus:

```text
current_density, pred_h{1..H}_<feature>  (H × 10 columns)
```

Warm-up frames before `input_len` history is available leave prediction
columns empty.

## Notes / honest limitations

- The estimator is trained to forecast *the same detector's feature stream
  one or more frames ahead*. R² values measure self-consistency over the
  forecast horizon, not absolute traffic-density accuracy.
- Without a road mask, occupancy / density / congestion include sky and
  off-road pixels — fine for relative comparisons across models on the
  same camera, less meaningful for cross-camera absolute claims.
- Live HLS streams have ~5–15s of inherent buffering latency. The pipeline
  flushes the CSV per row, so a `Ctrl+C` leaves a valid file.
