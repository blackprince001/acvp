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
| T-001 | [ ] Update `pyproject.toml` with all core dependencies | P0 | None | 30min | All dependencies from SPEC Section 15 listed correctly |
| T-002 | [ ] Update `.gitignore` for ML project artifacts | P0 | T-001 | 15min | Ignores checkpoints, logs, data, .env, IDE files |
| T-003 | [ ] Create directory structure per SPEC Section 2 | P0 | None | 30min | All directories from spec exist |
| T-004 | [ ] Create `__init__.py` files in all `src/` subpackages | P0 | T-003 | 15min | All packages importable |
| T-005 | [ ] Implement `src/utils/config.py` — YAML config loader with validation | P0 | T-004 | 2h | Loads YAML, merges with base, validates required fields, supports CLI overrides |
| T-006 | [ ] Implement `src/utils/logging.py` — Logging setup with loguru | P0 | T-004 | 30min | Configurable log levels, file + console output |
| T-007 | [ ] Implement `src/utils/types.py` — Shared type definitions and dataclasses | P0 | T-004 | 1h | DetectionResult, TrackingResult, FilteredResult, PredictionResult dataclasses defined |
| T-008 | [ ] Create base `configs/base.yaml` with all default values | P0 | T-005 | 1h | All default values from spec sections present and valid |
| T-009 | [ ] Create model config files under `configs/models/detection/` | P1 | T-008 | 1h | YAML files for yolov8n/s/m/l and yolov11n/s/m/l |
| T-010 | [ ] Create model config files under `configs/models/segmentation/` | P1 | T-008 | 1h | YAML files for yolov8n-seg/s-seg and yolov11n-seg/s-seg |
| T-011 | [ ] Create model config files under `configs/models/ml_estimator/` | P1 | T-008 | 1h | YAML files for lstm, gru, tcn |
| T-012 | [ ] Create tracker config files under `configs/tracking/` | P1 | T-008 | 30min | YAML files for botsort and bytetrack |
| T-013 | [ ] Create experiment config files under `configs/experiments/` | P1 | T-009, T-010, T-011, T-012 | 1h | baseline_detection, baseline_segmentation, predictive_lstm/gru/tcn, edge_deployment |
| T-014 | [ ] Set up pytest configuration (`pytest.ini` or `pyproject.toml` section) | P2 | T-003 | 15min | `pytest tests/` runs successfully (even with no tests yet) |
| T-015 | [ ] Create placeholder `main.py` with CLI argument parsing | P1 | T-005, T-006 | 1h | `python main.py --help` shows all available commands and options |

---

## Phase 1: Data Pipeline

| ID | Task | Priority | Dependencies | Effort | Acceptance Criteria |
|----|------|----------|-------------|--------|-------------------|
| T-016 | [ ] Implement `src/data/raw_loader.py` — Video and image sequence loading | P0 | T-004, T-006 | 2h | Supports MP4/AVI/MKV and image sequences, returns frame iterator with metadata |
| T-017 | [ ] Implement `src/data/preprocess.py` — Frame preprocessing pipeline | P0 | T-016 | 2h | Resizes, normalizes, extracts frames at target FPS, creates index file |
| T-018 | [ ] Implement `src/data/roboflow_segmentation.py` — Roboflow segmentation pipeline | P0 | T-017 | 4h | Runs Roboflow segmentation on frames, generates per-frame masks, generates static road mask, saves annotations |
| T-019 | [ ] Create metadata schema and validation for scene metadata | P1 | T-005 | 1h | `metadata.yaml` validated against schema, helpful error messages on invalid fields |
| T-020 | [ ] Implement `src/data/dataset.py` — `TrafficDetectionDataset` class | P0 | T-017, T-018 | 2h | Returns (image, {boxes, labels, image_id}) tuples, supports transforms |
| T-021 | [ ] Implement `src/data/dataset.py` — `TrafficSegmentationDataset` class | P0 | T-018, T-020 | 2h | Returns (image, {boxes, labels, masks, image_id}) tuples, supports transforms |
| T-022 | [ ] Implement `src/data/dataset.py` — `TelemetryDataset` class | P1 | T-020 | 2h | Returns (input_window, target_window) tuples for ML estimator training |
| T-023 | [ ] Implement `src/data/augmentations.py` — Training augmentations | P1 | T-020 | 2h | Flip, brightness/contrast, blur, rotation all implemented |
| T-024 | [ ] Implement `src/data/splits.py` — Train/val/test split management | P1 | T-019 | 2h | Scene-level split (not frame-level), stratified by scene type, saves manifests |
| T-025 | [ ] Create `scripts/01_preprocess_data.py` — Preprocessing entry point | P1 | T-016, T-017, T-019 | 1h | Loads config, runs preprocessing pipeline, outputs to `data/processed/` |
| T-026 | [ ] Create `scripts/02_run_roboflow_segmentation.py` — Roboflow segmentation entry point | P1 | T-018, T-025 | 1h | Loads config, runs Roboflow pipeline, outputs to `data/segmented/` |
| T-027 | [ ] Create split manifests for your dataset | P0 | T-024, Your dataset | 1h | `data/splits/train.txt`, `val.txt`, `test.txt` exist with correct scene assignments |
| T-028 | [ ] Write unit tests for `raw_loader.py` | P2 | T-016, T-014 | 1h | Tests video loading, frame extraction, error handling |
| T-029 | [ ] Write unit tests for `preprocess.py` | P2 | T-017, T-014 | 1h | Tests resizing, normalization, FPS extraction |
| T-030 | [ ] Write unit tests for `dataset.py` | P2 | T-020, T-021, T-014 | 1h | Tests **getitem** output shapes, transforms application |

---

## Phase 2: CV Model Training — Detection

| ID | Task | Priority | Dependencies | Effort | Acceptance Criteria |
|----|------|----------|-------------|--------|-------------------|
| T-031 | [ ] Implement `src/models/detection/base.py` — Base detection interface | P0 | T-004, T-007 | 1h | Abstract class with train(), predict(), export_onnx() methods |
| T-032 | [ ] Implement `src/models/detection/yolov8.py` — YOLOv8 detection wrapper | P0 | T-031, T-020 | 2h | Wraps Ultralytics YOLOv8, supports train/predict/export, loads from config |
| T-033 | [ ] Implement `src/models/detection/yolov11.py` — YOLO11 detection wrapper | P0 | T-031, T-020 | 2h | Wraps Ultralytics YOLO11, supports train/predict/export, loads from config |
| T-034 | [ ] Implement `src/models/detection/registry.py` — Model registry | P1 | T-032, T-033 | 1h | Config-based model loading: `registry.get_model("yolov8s")` returns correct model |
| T-035 | [ ] Create `scripts/03_train_detection.py` — Detection training entry point | P0 | T-032, T-033, T-034 | 2h | Loads config, trains model, saves checkpoints, logs to TensorBoard+W&B |
| T-036 | [ ] Train YOLOv8n detection model | P1 | T-027, T-035 | 2-4h | Model trained, checkpoint saved, metrics logged |
| T-037 | [ ] Train YOLOv8s detection model | P1 | T-027, T-035 | 2-4h | Model trained, checkpoint saved, metrics logged |
| T-038 | [ ] Train YOLOv8m detection model | P2 | T-027, T-035 | 4-6h | Model trained, checkpoint saved, metrics logged |
| T-039 | [ ] Train YOLOv8l detection model | P2 | T-027, T-035 | 4-6h | Model trained, checkpoint saved, metrics logged |
| T-040 | [ ] Train YOLO11n detection model | P1 | T-027, T-035 | 2-4h | Model trained, checkpoint saved, metrics logged |
| T-041 | [ ] Train YOLO11s detection model | P1 | T-027, T-035 | 2-4h | Model trained, checkpoint saved, metrics logged |
| T-042 | [ ] Train YOLO11m detection model | P2 | T-027, T-035 | 4-6h | Model trained, checkpoint saved, metrics logged |
| T-043 | [ ] Train YOLO11l detection model | P2 | T-027, T-035 | 4-6h | Model trained, checkpoint saved, metrics logged |
| T-044 | [ ] Write unit tests for detection models | P2 | T-032, T-033, T-014 | 1h | Tests forward pass, output shapes, config loading |

---

## Phase 3: CV Model Training — Segmentation

| ID | Task | Priority | Dependencies | Effort | Acceptance Criteria |
|----|------|----------|-------------|--------|-------------------|
| T-045 | [ ] Implement `src/models/segmentation/base.py` — Base segmentation interface | P0 | T-004, T-007 | 1h | Abstract class with train(), predict(), export_onnx() methods, mask output |
| T-046 | [ ] Implement `src/models/segmentation/yolov8_seg.py` — YOLOv8 segmentation wrapper | P0 | T-045, T-021 | 2h | Wraps Ultralytics YOLOv8-seg, supports train/predict/export with masks |
| T-047 | [ ] Implement `src/models/segmentation/yolov11_seg.py` — YOLO11 segmentation wrapper | P0 | T-045, T-021 | 2h | Wraps Ultralytics YOLO11-seg, supports train/predict/export with masks |
| T-048 | [ ] Update model registry to include segmentation models | P1 | T-034, T-046, T-047 | 30min | Registry returns correct segmentation models by name |
| T-049 | [ ] Create `scripts/04_train_segmentation.py` — Segmentation training entry point | P0 | T-046, T-047, T-048 | 2h | Loads config, trains seg model, saves checkpoints, logs to TensorBoard+W&B |
| T-050 | [ ] Train YOLOv8n-seg segmentation model | P1 | T-027, T-049 | 2-4h | Model trained, checkpoint saved, metrics logged |
| T-051 | [ ] Train YOLOv8s-seg segmentation model | P1 | T-027, T-049 | 2-4h | Model trained, checkpoint saved, metrics logged |
| T-052 | [ ] Train YOLO11n-seg segmentation model | P1 | T-027, T-049 | 2-4h | Model trained, checkpoint saved, metrics logged |
| T-053 | [ ] Train YOLO11s-seg segmentation model | P1 | T-027, T-049 | 2-4h | Model trained, checkpoint saved, metrics logged |
| T-054 | [ ] Write unit tests for segmentation models | P2 | T-046, T-047, T-014 | 1h | Tests forward pass, mask output shapes, config loading |

---

## Phase 4: Tracking Integration

| ID | Task | Priority | Dependencies | Effort | Acceptance Criteria |
|----|------|----------|-------------|--------|-------------------|
| T-055 | [ ] Implement `src/tracking/base.py` — Base tracker interface | P0 | T-004, T-007 | 1h | Abstract class with update(), get_trajectories() methods |
| T-056 | [ ] Implement `src/tracking/ultralytics_tracker.py` — Ultralytics tracking wrapper | P0 | T-055 | 2h | Wraps BoT-SORT and ByteTrack, returns tracked objects with IDs, supports both modes |
| T-057 | [ ] Implement movement vector calculation | P1 | T-056 | 2h | Computes velocity (m/s), direction (degrees), acceleration from tracked centroids |
| T-058 | [ ] Implement redundant counting prevention logic | P1 | T-056 | 1h | Zone-based counting, unique ID tracking per zone crossing |
| T-059 | [ ] Write unit tests for tracker | P2 | T-056, T-057, T-014 | 1h | Tests ID persistence, trajectory retrieval, movement vector computation |

---

## Phase 5: Spatial Logic Layer

| ID | Task | Priority | Dependencies | Effort | Acceptance Criteria |
|----|------|----------|-------------|--------|-------------------|
| T-060 | [ ] Implement `src/spatial_logic/road_mask.py` — Road mask manager | P0 | T-004, T-007 | 2h | Loads static road mask, generates from segmentation output, returns mask per frame |
| T-061 | [ ] Implement `src/spatial_logic/intersection.py` — IoU-based overlap computation | P0 | T-060 | 2h | Mask-road IoU, box-road IoU, both return overlap ratio in [0, 1] |
| T-062 | [ ] Implement `src/spatial_logic/filtering.py` — On-road classification | P0 | T-061 | 2h | Classifies vehicles as on-road/off-road based on overlap threshold + optional velocity filter |
| T-063 | [ ] Implement `src/spatial_logic/occupancy.py` — Pixel occupancy computation | P0 | T-060 | 1h | Computes road pixel occupancy ratio from vehicle masks |
| T-064 | [ ] Write unit tests for road_mask.py | P2 | T-060, T-014 | 1h | Tests mask loading, generation, merging |
| T-065 | [ ] Write unit tests for intersection.py | P2 | T-061, T-014 | 1h | Tests IoU computation with known inputs, edge cases (empty masks, full overlap) |
| T-066 | [ ] Write unit tests for filtering.py | P2 | T-062, T-014 | 1h | Tests on-road classification with synthetic tracked objects |

---

## Phase 6: Feature Engineering

| ID | Task | Priority | Dependencies | Effort | Acceptance Criteria |
|----|------|----------|-------------|--------|-------------------|
| T-067 | [ ] Implement `src/features/extractor.py` — Feature extraction from telemetry | P0 | T-057, T-062, T-063 | 2h | Extracts all 10 features per frame (counts, occupancy, velocity, direction, density, flow, congestion) |
| T-068 | [ ] Implement `src/features/windowing.py` — Temporal window builder | P0 | T-067 | 1h | Creates sliding windows (input: T-N..T, output: T+1..T+H) with configurable stride |
| T-069 | [ ] Implement `src/features/normalizer.py` — Feature normalization | P0 | T-067 | 1h | Supports standard, minmax, robust normalization; fit/transform/inverse_transform |
| T-070 | [ ] Generate telemetry data from CV pipeline on test scenes | P1 | T-036, T-050, T-056, T-060, T-062, T-067 | 2h | CSV/Parquet files in `data/generated/` with all features computed |
| T-071 | [ ] Write unit tests for feature extractor | P2 | T-067, T-014 | 1h | Tests feature computation with synthetic tracked objects |

---

## Phase 7: ML Estimator Training

| ID | Task | Priority | Dependencies | Effort | Acceptance Criteria |
|----|------|----------|-------------|--------|-------------------|
| T-072 | [ ] Implement `src/models/ml_estimator/base.py` — Base estimator interface | P0 | T-004 | 1h | Abstract class with forward() method, common training utilities |
| T-073 | [ ] Implement `src/models/ml_estimator/lstm.py` — LSTM architecture | P0 | T-072 | 2h | Multi-layer LSTM with FC head, configurable hidden size, dropout, bidirectional |
| T-074 | [ ] Implement `src/models/ml_estimator/gru.py` — GRU architecture | P0 | T-072 | 2h | Multi-layer GRU with FC head, configurable hidden size, dropout, bidirectional |
| T-075 | [ ] Implement `src/models/ml_estimator/tcn.py` — TCN architecture | P0 | T-072 | 3h | Temporal blocks with residual connections, dilated convolutions, FC head |
| T-076 | [ ] Create `scripts/05_train_ml_estimator.py` — ML estimator training entry point | P0 | T-073, T-074, T-075, T-068, T-069 | 3h | Loads config, trains model, saves checkpoints, logs to TensorBoard+W&B, early stopping |
| T-077 | [ ] Train LSTM estimator | P1 | T-070, T-076 | 2-4h | Model trained, checkpoint saved, metrics logged, convergence verified |
| T-078 | [ ] Train GRU estimator | P1 | T-070, T-076 | 2-4h | Model trained, checkpoint saved, metrics logged, convergence verified |
| T-079 | [ ] Train TCN estimator | P1 | T-070, T-076 | 2-4h | Model trained, checkpoint saved, metrics logged, convergence verified |
| T-080 | [ ] Write unit tests for ML estimator models | P2 | T-073, T-074, T-075, T-014 | 1h | Tests forward pass, output shapes, gradient flow |

---

## Phase 8: Inference Pipeline

| ID | Task | Priority | Dependencies | Effort | Acceptance Criteria |
|----|------|----------|-------------|--------|-------------------|
| T-081 | [ ] Implement `src/inference/queue.py` — Inter-component communication queues | P0 | T-007 | 1h | Thread-safe queue with overflow handling (drop_oldest, block, drop_newest policies) |
| T-082 | [ ] Implement `src/inference/video_io.py` — Video input/output handling | P0 | T-016 | 1h | Reads video frames, writes output video with overlays, handles different codecs |
| T-083 | [ ] Implement `src/inference/orchestrator.py` — Main pipeline orchestrator | P0 | T-032, T-033, T-046, T-047, T-056, T-060, T-062, T-067, T-073, T-074, T-075 | 4h | Coordinates all components, supports detect/segment/predict modes, outputs density stream |
| T-084 | [ ] Implement `src/inference/parallel.py` — Async/parallel execution engine | P1 | T-081, T-083 | 3h | Runs pipeline stages in parallel using threads/processes, manages queues between stages |
| T-085 | [ ] Implement fixed-FPS inference controller | P1 | T-083 | 1h | Maintains target FPS, sleeps if too fast, skips frames if too slow, logs actual FPS |
| T-086 | [ ] Create `scripts/06_run_inference.py` — Inference entry point | P0 | T-083, T-084 | 2h | Loads config, runs pipeline on video, outputs results (video overlay + telemetry data) |
| T-087 | [ ] Test end-to-end inference on short video clip | P1 | T-086 | 1h | Pipeline runs without errors, produces valid output |

---

## Phase 9: Model Optimization

| ID | Task | Priority | Dependencies | Effort | Acceptance Criteria |
|----|------|----------|-------------|--------|-------------------|
| T-088 | [ ] Implement `src/optimization/onnx_export.py` — ONNX export utilities | P1 | T-032, T-033, T-046, T-047 | 2h | Exports detection and segmentation models to ONNX with dynamic axes |
| T-089 | [ ] Implement `src/optimization/tensorrt.py` — TensorRT conversion | P2 | T-088 | 3h | Converts ONNX to TensorRT engine, supports FP16 and INT8 |
| T-090 | [ ] Implement `src/optimization/benchmark.py` — Optimization benchmarking | P1 | T-088, T-089 | 2h | Measures latency (mean, p50, p95, p99), FPS, GPU memory for each format |
| T-091 | [ ] Export all trained detection models to ONNX | P2 | T-036-T-043, T-088 | 1h | ONNX files saved, validated with onnxruntime |
| T-092 | [ ] Export all trained segmentation models to ONNX | P2 | T-050-T-053, T-088 | 1h | ONNX files saved, validated with onnxruntime |
| T-093 | [ ] Create `scripts/08_export_models.py` — Model export entry point | P2 | T-088, T-089 | 1h | CLI tool to export models to ONNX/TensorRT with configurable precision |
| T-094 | [ ] Benchmark ONNX models vs PyTorch | P2 | T-091, T-092, T-090 | 1h | Results logged, speedup factors computed |
| T-095 | [ ] Benchmark TensorRT models vs ONNX (if TensorRT available) | P3 | T-089, T-090 | 1h | Results logged, speedup factors computed |

---

## Phase 10: Evaluation & Benchmarking

| ID | Task | Priority | Dependencies | Effort | Acceptance Criteria |
|----|------|----------|-------------|--------|-------------------|
| T-096 | [ ] Implement `src/evaluation/metrics.py` — All metric implementations | P0 | T-007 | 2h | FPS, latency, MAE, RMSE, MAPE, R², detection metrics (precision, recall, mAP) |
| T-097 | [ ] Implement `src/evaluation/benchmark.py` — Benchmark runner | P0 | T-096, T-083 | 3h | Runs model benchmarks, predictor benchmarks, environmental comparison |
| T-098 | [ ] Implement `src/evaluation/visualization.py` — Graph generation | P1 | T-096 | 3h | All plot functions from SPEC Section 11.3 implemented, publication-quality output |
| T-099 | [ ] Run FPS benchmark across all YOLO detection variants | P1 | T-036-T-043, T-097 | 2h | Results saved, FPS comparison data available |
| T-100 | [ ] Run latency benchmark across all YOLO detection variants | P1 | T-036-T-043, T-097 | 2h | Results saved, latency distribution data available |
| T-101 | [ ] Run MAE benchmark (vehicle count accuracy) across all variants | P1 | T-036-T-043, T-050-T-053, T-097 | 2h | Results saved, MAE comparison data available |
| T-102 | [ ] Run detection vs segmentation accuracy trade-off analysis | P1 | T-099, T-100, T-101 | 1h | Results saved, trade-off data available |
| T-103 | [ ] Run ML estimator accuracy benchmark (RMSE, MAPE, R²) | P1 | T-077-T-079, T-097 | 2h | Results saved, predictor comparison data available |
| T-104 | [ ] Run environmental comparison (highway vs urban vs suburban) | P1 | T-099, T-101, T-097 | 2h | Results saved, environment breakdown data available |
| T-105 | [ ] Run optimization impact benchmark (PyTorch vs ONNX vs TensorRT) | P2 | T-094, T-095, T-097 | 1h | Results saved, optimization impact data available |
| T-106 | [ ] Create `scripts/07_evaluate.py` — Evaluation entry point | P1 | T-097, T-098 | 1h | CLI tool to run all benchmarks and generate results |

---

## Phase 11: Experiment Tracking

| ID | Task | Priority | Dependencies | Effort | Acceptance Criteria |
|----|------|----------|-------------|--------|-------------------|
| T-107 | [ ] Implement `src/experiment/tracker.py` — Unified experiment tracker | P1 | T-005 | 2h | Supports multiple backends, unified API for logging params, metrics, images, models |
| T-108 | [ ] Implement `src/experiment/tensorboard_logger.py` — TensorBoard integration | P1 | T-107 | 1h | Logs scalars, images, histograms, graphs to TensorBoard |
| T-109 | [ ] Implement `src/experiment/wandb_logger.py` — W&B integration | P1 | T-107 | 1h | Logs all metric types, model artifacts, supports offline mode |
| T-110 | [ ] Integrate experiment tracking into detection training script | P1 | T-107, T-035 | 1h | Training logs to both TensorBoard and W&B |
| T-111 | [ ] Integrate experiment tracking into segmentation training script | P1 | T-107, T-049 | 1h | Training logs to both TensorBoard and W&B |
| T-112 | [ ] Integrate experiment tracking into ML estimator training script | P1 | T-107, T-076 | 1h | Training logs to both TensorBoard and W&B |
| T-113 | [ ] Integrate experiment tracking into benchmark runner | P1 | T-107, T-097 | 1h | Benchmark results logged to both backends |

---

## Phase 12: Paper Infrastructure

| ID | Task | Priority | Dependencies | Effort | Acceptance Criteria |
|----|------|----------|-------------|--------|-------------------|
| T-114 | [ ] Set up `paper/figures/` directory structure | P2 | None | 15min | All subdirectories exist (architecture, performance, comparison, environment) |
| T-115 | [ ] Set up `paper/tables/` directory structure | P2 | None | 15min | Directory exists for CSV/JSON table data |
| T-116 | [ ] Create `paper/references.bib` — Bibliography file | P2 | None | 1h | BibTeX entries for YOLO, tracking, Roboflow, density estimation papers |
| T-117 | [ ] Generate architecture diagram (Figure 1) | P2 | T-098 | 1h | SVG diagram of system architecture saved to `paper/figures/architecture/` |
| T-118 | [ ] Generate pipeline data flow diagram (Figure 2) | P2 | T-098 | 1h | SVG diagram of data flow saved to `paper/figures/architecture/` |
| T-119 | [ ] Generate FPS comparison chart (Figure 3) | P2 | T-099, T-098 | 30min | Bar chart saved to `paper/figures/performance/` |
| T-120 | [ ] Generate latency distribution box plots (Figure 4) | P2 | T-100, T-098 | 30min | Box plots saved to `paper/figures/performance/` |
| T-121 | [ ] Generate MAE comparison chart (Figure 5) | P2 | T-101, T-098 | 30min | Bar chart saved to `paper/figures/performance/` |
| T-122 | [ ] Generate detection vs segmentation trade-off chart (Figure 6) | P2 | T-102, T-098 | 30min | Scatter/line chart saved to `paper/figures/comparison/` |
| T-123 | [ ] Generate actual vs predicted density time series (Figure 7) | P2 | T-103, T-098 | 30min | Time series plot saved to `paper/figures/performance/` |
| T-124 | [ ] Generate RMSE/MAPE across LSTM/GRU/TCN chart (Figure 8) | P2 | T-103, T-098 | 30min | Grouped bar chart saved to `paper/figures/comparison/` |
| T-125 | [ ] Generate environment comparison chart (Figure 9) | P2 | T-104, T-098 | 30min | Grouped bar chart saved to `paper/figures/environment/` |
| T-126 | [ ] Generate optimization impact chart (Figure 10) | P2 | T-105, T-098 | 30min | Before/after bar chart saved to `paper/figures/performance/` |
| T-127 | [ ] Export all table data (Tables 1-8) | P2 | T-099-T-105 | 1h | CSV/JSON files for all 8 tables saved to `paper/tables/` |
| T-128 | [ ] Create `scripts/09_generate_paper_figures.py` — Figure generation entry point | P2 | T-117-T-127 | 1h | CLI tool to regenerate all figures from results data |

---

## Phase 13: Vision Transformer Segmentation (Future Optimization)

| ID | Task | Priority | Dependencies | Effort | Acceptance Criteria |
|----|------|----------|-------------|--------|-------------------|
| T-129 | [ ] Research and select ViT segmentation architecture | P3 | None | 2h | Architecture decision documented (ViT + Mask2Former or SETR) |
| T-130 | [ ] Implement `src/models/segmentation/vit_seg.py` — ViT segmentation model | P3 | T-129 | 4h | ViT-based segmentation model with configurable backbone and decoder |
| T-131 | [ ] Train ViT segmentation model | P3 | T-130, T-027 | 6-12h | Model trained, checkpoint saved, metrics logged |
| T-132 | [ ] Benchmark ViT segmentation vs YOLO segmentation | P3 | T-131, T-097 | 1h | Results saved, accuracy/speed comparison available |
| T-133 | [ ] Write unit tests for ViT segmentation model | P3 | T-130, T-014 | 1h | Tests forward pass, mask output shapes |

---

## Phase 14: Documentation & Final Review

| ID | Task | Priority | Dependencies | Effort | Acceptance Criteria |
|----|------|----------|-------------|--------|-------------------|
| T-134 | [ ] Write comprehensive README.md | P1 | All phases complete | 2h | Project overview, setup instructions, usage examples, architecture description |
| T-135 | [ ] Add docstrings to all public classes and functions | P1 | All phases complete | 4h | Every public API has docstring with args, returns, examples |
| T-136 | [ ] Run full test suite and fix any failures | P0 | All phases complete | 2h | `pytest tests/ -v` passes with 100% pass rate |
| T-137 | [ ] Run linting and fix style issues | P2 | All phases complete | 1h | No linting errors |
| T-138 | [ ] Verify all experiment configs are valid and runnable | P1 | All phases complete | 1h | Each config file loads without errors |
| T-139 | [ ] Verify end-to-end pipeline on test video | P0 | All phases complete | 1h | Full pipeline (detect → track → filter → predict) runs on test video |
| T-140 | [ ] Verify all paper figures are generated correctly | P2 | T-128 | 30min | All 10 figures exist and are publication-quality |
| T-141 | [ ] Cross-reference TASKS.md with SPEC.md for completeness | P1 | All phases complete | 1h | Every spec section has corresponding tasks |
| T-142 | [ ] Final review of PROBLEM.md for accuracy and completeness | P1 | All phases complete | 30min | Problem statement reflects actual implementation |

---

## Execution Order Summary

```
Phase 0  ──▶ Phase 1  ──▶ Phase 2 ──┐
                                     ├──▶ Phase 4 ──▶ Phase 5 ──▶ Phase 6 ──▶ Phase 7
                          Phase 3 ──┘                                              │
                                                                                   ▼
Phase 11 (parallel)                                                    Phase 8 ──▶ Phase 9
                                                                                   │
                                                                                   ▼
Phase 14 ◀── Phase 13 ◀── Phase 12 ◀─────────────────────────────── Phase 10
```

**Parallel tracks:**

- Phase 11 (Experiment Tracking) can begin as soon as Phase 0 is complete
- Phase 13 (ViT) can begin after Phase 3, runs parallel to Phases 4-10
- Phase 12 (Paper) begins after Phase 10 results are available

---

## Task Statistics

| Category | Count |
|----------|-------|
| Total tasks | 142 |
| P0 (Critical) | 38 |
| P1 (High) | 52 |
| P2 (Medium) | 44 |
| P3 (Low/Future) | 8 |

| Phase | Tasks | Estimated Total Effort |
|-------|-------|----------------------|
| Phase 0: Scaffolding | 15 | ~12h |
| Phase 1: Data Pipeline | 15 | ~22h |
| Phase 2: Detection Training | 14 | ~25h (mostly GPU time) |
| Phase 3: Segmentation Training | 10 | ~18h (mostly GPU time) |
| Phase 4: Tracking | 5 | ~6h |
| Phase 5: Spatial Logic | 7 | ~9h |
| Phase 6: Feature Engineering | 5 | ~6h |
| Phase 7: ML Estimator | 9 | ~18h (mostly GPU time) |
| Phase 8: Inference Pipeline | 7 | ~14h |
| Phase 9: Optimization | 8 | ~10h |
| Phase 10: Evaluation | 11 | ~16h |
| Phase 11: Experiment Tracking | 7 | ~8h |
| Phase 12: Paper Infrastructure | 15 | ~8h |
| Phase 13: ViT Segmentation | 5 | ~18h |
| Phase 14: Documentation | 9 | ~12h |
