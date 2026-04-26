# Task List: Real-Time Traffic Density Estimation

## Legend

| Symbol | Meaning |
|--------|---------|
| `[ ]` | Not started |
| `[~]` | In progress |
| `[x]` | Completed |
| `[!]` | Blocked |

**Priority:** P0 = Critical, P1 = High, P2 = Medium, P3 = Low

---

## Phase 0: Project Scaffolding

| ID | Task | Priority | Dependencies | Effort | Acceptance Criteria |
|----|------|----------|-------------|--------|-------------------|
| T-001 | [x] Update `pyproject.toml` with all core dependencies | P0 | None | 30min | All dependencies from SPEC Section 15 listed correctly |
| T-002 | [x] Update `.gitignore` for ML project artifacts | P0 | T-001 | 15min | Ignores checkpoints, logs, data, .env, IDE files |
| T-003 | [x] Create directory structure per SPEC Section 2 | P0 | None | 30min | All directories from spec exist |
| T-004 | [x] Create `__init__.py` files in all `src/` subpackages | P0 | T-003 | 15min | All packages importable |
| T-005 | [x] Implement `src/utils/config.py` — YAML config loader with validation | P0 | T-004 | 2h | Loads YAML, merges with base, validates required fields, supports CLI overrides |
| T-006 | [x] Implement `src/utils/logging.py` — Logging setup with loguru | P0 | T-004 | 30min | Configurable log levels, file + console output |
| T-007 | [x] Implement `src/utils/types.py` — Shared type definitions and dataclasses | P0 | T-004 | 1h | DetectionResult, TrackingResult, FilteredResult, PredictionResult dataclasses defined |
| T-008 | [x] Create base `configs/base.yaml` with all default values | P0 | T-005 | 1h | All default values from spec sections present and valid |
| T-009 | [x] Create model config files under `configs/models/detection/` | P1 | T-008 | 1h | YAML files for yolov8n/s/m/l and yolov11n/s/m/l |
| T-010 | [x] Create model config files under `configs/models/segmentation/` | P1 | T-008 | 1h | YAML files for yolov8n-seg/s-seg and yolov11n-seg/s-seg |
| T-011 | [x] Create model config files under `configs/models/ml_estimator/` | P1 | T-008 | 1h | YAML files for lstm, gru, tcn |
| T-012 | [x] Create tracker config files under `configs/tracking/` | P1 | T-008 | 30min | YAML files for botsort and bytetrack |
| T-013 | [x] Create experiment config files under `configs/experiments/` | P1 | T-009, T-010, T-011, T-012 | 1h | baseline_detection, baseline_segmentation, predictive_lstm/gru/tcn, edge_deployment |
| T-014 | [x] Set up pytest configuration (`pytest.ini` or `pyproject.toml` section) | P2 | T-003 | 15min | `pytest tests/` runs successfully (even with no tests yet) |
| T-015 | [x] Create placeholder `main.py` with CLI argument parsing | P1 | T-005, T-006 | 1h | `python main.py --help` shows all available commands and options |

---

## Phase 1: Data Pipeline

| ID | Task | Priority | Dependencies | Effort | Acceptance Criteria |
|----|------|----------|-------------|--------|-------------------|
| T-016 | [x] Implement `src/data/raw_loader.py` — Video and image sequence loading | P0 | T-004, T-006 | 2h | Supports MP4/AVI/MKV and image sequences, returns frame iterator with metadata |
| T-017 | [x] Implement `src/data/preprocess.py` — Frame preprocessing pipeline | P0 | T-016 | 2h | Resizes, normalizes, extracts frames at target FPS, creates index file |
| T-018 | [x] Implement `src/data/roboflow_segmentation.py` — Roboflow segmentation pipeline | P0 | T-017 | 4h | Runs Roboflow segmentation on frames, generates per-frame masks, generates static road mask, saves annotations |
| T-019 | [x] Create metadata schema and validation for scene metadata | P1 | T-005 | 1h | `metadata.yaml` validated against schema, helpful error messages on invalid fields |
| T-020 | [x] Implement `src/data/dataset.py` — `TrafficDetectionDataset` class | P0 | T-017, T-018 | 2h | Returns (image, {boxes, labels, image_id}) tuples, supports transforms |
| T-021 | [x] Implement `src/data/dataset.py` — `TrafficSegmentationDataset` class | P0 | T-018, T-020 | 2h | Returns (image, {boxes, labels, masks, image_id}) tuples, supports transforms |
| T-022 | [x] Implement `src/data/dataset.py` — `TelemetryDataset` class | P1 | T-020 | 2h | Returns (input_window, target_window) tuples for ML estimator training |
| T-023 | [x] Implement `src/data/augmentations.py` — Training augmentations | P1 | T-020 | 2h | Flip, brightness/contrast, blur, rotation all implemented |
| T-024 | [x] Implement `src/data/splits.py` — Train/val/test split management | P1 | T-019 | 2h | Scene-level split (not frame-level), stratified by scene type, saves manifests |
| T-025 | [x] Create `scripts/01_preprocess_data.py` — Preprocessing entry point | P1 | T-016, T-017, T-019 | 1h | Loads config, runs preprocessing pipeline, outputs to `data/processed/` |
| T-026 | [x] Create `scripts/02_run_roboflow_segmentation.py` — Roboflow segmentation entry point | P1 | T-018, T-025 | 1h | Loads config, runs Roboflow pipeline, outputs to `data/segmented/` |
| T-027 | [x] Create split manifests for your dataset | P0 | T-024, Your dataset | 1h | `data/splits/train.txt`, `val.txt`, `test.txt` exist with correct scene assignments |
| T-028 | [x] Write unit tests for `raw_loader.py` | P2 | T-016, T-014 | 1h | Tests video loading, frame extraction, error handling |
| T-029 | [x] Write unit tests for `preprocess.py` | P2 | T-017, T-014 | 1h | Tests resizing, normalization, FPS extraction |
| T-030 | [x] Write unit tests for `dataset.py` | P2 | T-020, T-021, T-014 | 1h | Tests **getitem** output shapes, transforms application |

---

## Phase 2: CV Model Training — Detection

| ID | Task | Priority | Dependencies | Effort | Acceptance Criteria |
|----|------|----------|-------------|--------|-------------------|
| T-031 | [x] Implement `src/models/detection/base.py` — Base detection interface | P0 | T-004, T-007 | 1h | Abstract class with train(), predict(), export_onnx() methods |
| T-032 | [x] Implement `src/models/detection/yolov8.py` — YOLOv8 detection wrapper | P0 | T-031, T-020 | 2h | Wraps Ultralytics YOLOv8, supports train/predict/export, loads from config |
| T-033 | [x] Implement `src/models/detection/yolov11.py` — YOLO11 detection wrapper | P0 | T-031, T-020 | 2h | Wraps Ultralytics YOLO11, supports train/predict/export, loads from config |
| T-034 | [x] Implement `src/models/detection/registry.py` — Model registry | P1 | T-032, T-033 | 1h | Config-based model loading: `registry.get_model("yolov8s")` returns correct model |
| T-035 | [x] Create `scripts/03_train_detection.py` — Detection training entry point | P0 | T-032, T-033, T-034 | 2h | Loads config, trains model, saves checkpoints, logs to TensorBoard+W&B |
| T-036 | [ ] Train YOLOv8n detection model | P1 | T-027, T-035 | 2-4h | Model trained, checkpoint saved, metrics logged |
| T-037 | [ ] Train YOLOv8s detection model | P1 | T-027, T-035 | 2-4h | Model trained, checkpoint saved, metrics logged |
| T-038 | [ ] Train YOLOv8m detection model | P2 | T-027, T-035 | 4-6h | Model trained, checkpoint saved, metrics logged |
| T-039 | [ ] Train YOLOv8l detection model | P2 | T-027, T-035 | 4-6h | Model trained, checkpoint saved, metrics logged |
| T-040 | [ ] Train YOLO11n detection model | P1 | T-027, T-035 | 2-4h | Model trained, checkpoint saved, metrics logged |
| T-041 | [ ] Train YOLO11s detection model | P1 | T-027, T-035 | 2-4h | Model trained, checkpoint saved, metrics logged |
| T-042 | [ ] Train YOLO11m detection model | P2 | T-027, T-035 | 4-6h | Model trained, checkpoint saved, metrics logged |
| T-043 | [ ] Train YOLO11l detection model | P2 | T-027, T-035 | 4-6h | Model trained, checkpoint saved, metrics logged |
| T-044 | [x] Write unit tests for detection models | P2 | T-032, T-033, T-014 | 1h | Tests forward pass, output shapes, config loading |

---

---

## Phase 3: Tracking Integration

| ID | Task | Priority | Dependencies | Effort | Acceptance Criteria |
|----|------|----------|-------------|--------|-------------------|
| T-045 | [x] Implement `src/tracking/base.py` — Base tracker interface | P0 | T-004, T-007 | 1h | Abstract class with update(), get_trajectories() methods |
| T-046 | [x] Implement `src/tracking/ultralytics_tracker.py` — Ultralytics tracking wrapper | P0 | T-045 | 2h | Wraps BoT-SORT and ByteTrack, returns tracked objects with IDs, supports both modes |
| T-047 | [x] Implement movement vector calculation | P1 | T-046 | 2h | Computes velocity (m/s), direction (degrees), acceleration from tracked centroids |
| T-048 | [x] Implement redundant counting prevention logic | P1 | T-046 | 1h | Zone-based counting, unique ID tracking per zone crossing |
| T-049 | [x] Write unit tests for tracker | P2 | T-046, T-047, T-014 | 1h | Tests ID persistence, trajectory retrieval, movement vector computation |

---

## Phase 4: Spatial Logic Layer

| ID | Task | Priority | Dependencies | Effort | Acceptance Criteria |
|----|------|----------|-------------|--------|-------------------|
| T-050 | [x] Implement `src/spatial_logic/road_mask.py` — Road mask manager | P0 | T-004, T-007 | 2h | Loads static road mask, generates from segmentation output, returns mask per frame |
| T-051 | [x] Implement `src/spatial_logic/intersection.py` — IoU-based overlap computation | P0 | T-050 | 2h | Mask-road IoU, box-road IoU, both return overlap ratio in [0, 1] |
| T-052 | [x] Implement `src/spatial_logic/filtering.py` — On-road classification | P0 | T-051 | 2h | Classifies vehicles as on-road/off-road based on overlap threshold + optional velocity filter |
| T-053 | [x] Implement `src/spatial_logic/occupancy.py` — Pixel occupancy computation | P0 | T-050 | 1h | Computes road pixel occupancy ratio from vehicle masks |
| T-054 | [x] Write unit tests for road_mask.py | P2 | T-050, T-014 | 1h | Tests mask loading, generation, merging |
| T-055 | [x] Write unit tests for intersection.py | P2 | T-051, T-014 | 1h | Tests IoU computation with known inputs, edge cases (empty masks, full overlap) |
| T-056 | [x] Write unit tests for filtering.py | P2 | T-052, T-014 | 1h | Tests on-road classification with synthetic tracked objects |

---

## Phase 5: Feature Engineering

| ID | Task | Priority | Dependencies | Effort | Acceptance Criteria |
|----|------|----------|-------------|--------|-------------------|
| T-057 | [x] Implement `src/features/extractor.py` — Feature extraction from telemetry | P0 | T-048, T-052, T-053 | 2h | Extracts all 10 features per frame (counts, occupancy, velocity, direction, density, flow, congestion) |
| T-058 | [x] Implement `src/features/windowing.py` — Temporal window builder | P0 | T-057 | 1h | Creates sliding windows (input: T-N..T, output: T+1..T+H) with configurable stride |
| T-059 | [x] Implement `src/features/normalizer.py` — Feature normalization | P0 | T-057 | 1h | Supports standard, minmax, robust normalization; fit/transform/inverse_transform |
| T-060 | [ ] Generate telemetry data from CV pipeline on test scenes | P1 | T-036, T-045, T-050, T-051, T-057 | 2h | CSV/Parquet files in `data/generated/` with all features computed |
| T-061 | [x] Write unit tests for feature extractor | P2 | T-057, T-014 | 1h | Tests feature computation with synthetic tracked objects |

---

## Phase 6: ML Estimator Training

| ID | Task | Priority | Dependencies | Effort | Acceptance Criteria |
|----|------|----------|-------------|--------|-------------------|
| T-062 | [x] Implement `src/models/ml_estimator/base.py` — Base estimator interface | P0 | T-004 | 1h | Abstract class with forward() method, common training utilities |
| T-063 | [x] Implement `src/models/ml_estimator/lstm.py` — LSTM architecture | P0 | T-062 | 2h | Multi-layer LSTM with FC head, configurable hidden size, dropout, bidirectional |
| T-064 | [x] Implement `src/models/ml_estimator/gru.py` — GRU architecture | P0 | T-062 | 2h | Multi-layer GRU with FC head, configurable hidden size, dropout, bidirectional |
| T-065 | [x] Implement `src/models/ml_estimator/tcn.py` — TCN architecture | P0 | T-062 | 3h | Temporal blocks with residual connections, dilated convolutions, FC head |
| T-066 | [ ] Create `scripts/04_train_ml_estimator.py` — ML estimator training entry point | P0 | T-063, T-064, T-065, T-058, T-059 | 3h | Loads config, trains model, saves checkpoints, logs to TensorBoard+W&B, early stopping |
| T-067 | [ ] Train LSTM estimator | P1 | T-060, T-066 | 2-4h | Model trained, checkpoint saved, metrics logged, convergence verified |
| T-068 | [ ] Train GRU estimator | P1 | T-060, T-066 | 2-4h | Model trained, checkpoint saved, metrics logged, convergence verified |
| T-069 | [ ] Train TCN estimator | P1 | T-060, T-066 | 2-4h | Model trained, checkpoint saved, metrics logged, convergence verified |
| T-070 | [x] Write unit tests for ML estimator models | P2 | T-063, T-064, T-065, T-014 | 1h | Tests forward pass, output shapes, gradient flow |

---

## Phase 7: Inference Pipeline

| ID | Task | Priority | Dependencies | Effort | Acceptance Criteria |
|----|------|----------|-------------|--------|-------------------|
| T-071 | [x] Implement `src/inference/queue.py` — Inter-component communication queues | P0 | T-007 | 1h | Thread-safe queue with overflow handling (drop_oldest, block, drop_newest policies) |
| T-072 | [x] Implement `src/inference/video_io.py` — Video input/output handling | P0 | T-016 | 1h | Reads video frames, writes output video with overlays, handles different codecs |
| T-073 | [x] Implement `src/inference/orchestrator.py` — Main pipeline orchestrator | P0 | T-032, T-033, T-045, T-046, T-050, T-051, T-052, T-057, T-062, T-063, T-064 | 4h | Coordinates all components, supports detect/predict modes, outputs density stream |
| T-074 | [x] Implement `src/inference/parallel.py` — Async/parallel execution engine | P1 | T-071, T-073 | 3h | Runs pipeline stages in parallel using threads/processes, manages queues between stages |
| T-075 | [x] Implement fixed-FPS inference controller | P1 | T-073 | 1h | Maintains target FPS, sleeps if too fast, skips frames if too slow, logs actual FPS |
| T-076 | [x] Create `scripts/05_run_inference.py` — Inference entry point | P0 | T-073, T-074 | 2h | Loads config, runs pipeline on video, outputs results (video overlay + telemetry data) |
| T-077 | [ ] Test end-to-end inference on short video clip | P1 | T-076 | 1h | Pipeline runs without errors, produces valid output |

---

## Phase 8: Model Optimization — DROPPED

Out of scope for this project. ONNX/TensorRT export and the associated
benchmarks (former T-078–T-084) will not be pursued. T-093 in Phase 9
and T-112 in Phase 11 (which depended on this phase) are also dropped.

---

## Phase 9: Evaluation & Benchmarking

| ID | Task | Priority | Dependencies | Effort | Acceptance Criteria |
|----|------|----------|-------------|--------|-------------------|
| T-085 | [x] Implement `src/evaluation/metrics.py` — All metric implementations | P0 | T-007 | 2h | FPS, latency, MAE, RMSE, MAPE, R², detection metrics (precision, recall, mAP) |
| T-086 | [x] Implement `src/evaluation/benchmark.py` — Benchmark runner (model-only, paper-style) | P0 | T-085 | 3h | Runs detection model benchmarks (val + latency loop), emits per-model JSON + summary CSV |
| T-087 | [x] Implement `src/evaluation/visualization.py` — Graph generation | P1 | T-085 | 3h | mAP/FPS bars, latency boxplot, speed-vs-accuracy scatter, size-vs-accuracy scatter, per-class AP heatmap |
| T-088 | [ ] Run FPS benchmark across all YOLO detection variants | P1 | T-036-T-043, T-086 | 2h | Results saved, FPS comparison data available |
| T-089 | [ ] Run latency benchmark across all YOLO detection variants | P1 | T-036-T-043, T-086 | 2h | Results saved, latency distribution data available |
| T-090 | [ ] Run MAE benchmark (vehicle count accuracy) across all variants | P1 | T-036-T-043, T-086 | 2h | Results saved, MAE comparison data available |
| T-091 | [ ] Run ML estimator accuracy benchmark (RMSE, MAPE, R²) | P1 | T-067-T-069, T-086 | 2h | Results saved, predictor comparison data available |
| T-092 | [ ] Run environmental comparison (highway vs urban vs suburban) | P1 | T-088, T-090, T-086 | 2h | Results saved, environment breakdown data available |
| T-093 | ~~Run optimization impact benchmark~~ — DROPPED with Phase 8 | — | — | — | — |
| T-094 | [x] Create `scripts/evaluate.py` — Evaluation entry point | P1 | T-086, T-087 | 1h | CLI tool to discover trained checkpoints, run benchmarks, render figures, log to tracker |

---

## Phase 10: Experiment Tracking

| ID | Task | Priority | Dependencies | Effort | Acceptance Criteria |
|----|------|----------|-------------|--------|-------------------|
| T-095 | [x] Implement `src/experiment/tracker.py` — Unified experiment tracker | P1 | T-005 | 2h | BaseLogger interface + ExperimentTracker fan-out + NoOpLogger; from_config builder |
| T-096 | [x] Implement `src/experiment/tensorboard_logger.py` — TensorBoard integration | P1 | T-095 | 1h | Wraps SummaryWriter; scalars + image artifacts + hparam logging |
| T-097 | [x] Implement `src/experiment/wandb_logger.py` — W&B integration | P1 | T-095 | 1h | Wraps wandb SDK; metrics + artifacts; supports offline/disabled modes |
| T-098 | [ ] Integrate experiment tracking into detection training script | P1 | T-095, T-035 | 1h | Training logs to both TensorBoard and W&B |
| T-099 | [ ] Integrate experiment tracking into ML estimator training script | P1 | T-095, T-066 | 1h | Training logs to both TensorBoard and W&B |
| T-100 | [x] Integrate experiment tracking into benchmark runner | P1 | T-095, T-086 | 1h | DetectionBenchmark logs params + per-model metrics + figure artifacts via ExperimentTracker |
