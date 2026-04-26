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

## Phase 8: Model Optimization

| ID | Task | Priority | Dependencies | Effort | Acceptance Criteria |
|----|------|----------|-------------|--------|-------------------|
| T-078 | [ ] Implement `src/optimization/onnx_export.py` — ONNX export utilities | P1 | T-032, T-033 | 2h | Exports detection models to ONNX with dynamic axes |
| T-079 | [ ] Implement `src/optimization/tensorrt.py` — TensorRT conversion | P2 | T-078 | 3h | Converts ONNX to TensorRT engine, supports FP16 and INT8 |
| T-080 | [ ] Implement `src/optimization/benchmark.py` — Optimization benchmarking | P1 | T-078, T-079 | 2h | Measures latency (mean, p50, p95, p99), FPS, GPU memory for each format |
| T-081 | [ ] Export all trained detection models to ONNX | P2 | T-036-T-043, T-078 | 1h | ONNX files saved, validated with onnxruntime |
| T-082 | [ ] Create `scripts/07_export_models.py` — Model export entry point | P2 | T-078, T-079 | 1h | CLI tool to export models to ONNX/TensorRT with configurable precision |
| T-083 | [ ] Benchmark ONNX models vs PyTorch | P2 | T-081, T-080 | 1h | Results logged, speedup factors computed |
| T-084 | [ ] Benchmark TensorRT models vs ONNX (if TensorRT available) | P3 | T-079, T-080 | 1h | Results logged, speedup factors computed |

---

## Phase 9: Evaluation & Benchmarking

| ID | Task | Priority | Dependencies | Effort | Acceptance Criteria |
|----|------|----------|-------------|--------|-------------------|
| T-085 | [ ] Implement `src/evaluation/metrics.py` — All metric implementations | P0 | T-007 | 2h | FPS, latency, MAE, RMSE, MAPE, R², detection metrics (precision, recall, mAP) |
| T-086 | [ ] Implement `src/evaluation/benchmark.py` — Benchmark runner | P0 | T-085, T-073 | 3h | Runs model benchmarks, predictor benchmarks, environmental comparison |
| T-087 | [ ] Implement `src/evaluation/visualization.py` — Graph generation | P1 | T-085 | 3h | All plot functions from SPEC Section 11.3 implemented, publication-quality output |
| T-088 | [ ] Run FPS benchmark across all YOLO detection variants | P1 | T-036-T-043, T-086 | 2h | Results saved, FPS comparison data available |
| T-089 | [ ] Run latency benchmark across all YOLO detection variants | P1 | T-036-T-043, T-086 | 2h | Results saved, latency distribution data available |
| T-090 | [ ] Run MAE benchmark (vehicle count accuracy) across all variants | P1 | T-036-T-043, T-086 | 2h | Results saved, MAE comparison data available |
| T-091 | [ ] Run ML estimator accuracy benchmark (RMSE, MAPE, R²) | P1 | T-067-T-069, T-086 | 2h | Results saved, predictor comparison data available |
| T-092 | [ ] Run environmental comparison (highway vs urban vs suburban) | P1 | T-088, T-090, T-086 | 2h | Results saved, environment breakdown data available |
| T-093 | [ ] Run optimization impact benchmark (PyTorch vs ONNX vs TensorRT) | P2 | T-083, T-084, T-086 | 1h | Results saved, optimization impact data available |
| T-094 | [ ] Create `scripts/06_evaluate.py` — Evaluation entry point | P1 | T-086, T-087 | 1h | CLI tool to run all benchmarks and generate results |

---

## Phase 10: Experiment Tracking

| ID | Task | Priority | Dependencies | Effort | Acceptance Criteria |
|----|------|----------|-------------|--------|-------------------|
| T-095 | [ ] Implement `src/experiment/tracker.py` — Unified experiment tracker | P1 | T-005 | 2h | Supports multiple backends, unified API for logging params, metrics, images, models |
| T-096 | [ ] Implement `src/experiment/tensorboard_logger.py` — TensorBoard integration | P1 | T-095 | 1h | Logs scalars, images, histograms, graphs to TensorBoard |
| T-097 | [ ] Implement `src/experiment/wandb_logger.py` — W&B integration | P1 | T-095 | 1h | Logs all metric types, model artifacts, supports offline mode |
| T-098 | [ ] Integrate experiment tracking into detection training script | P1 | T-095, T-035 | 1h | Training logs to both TensorBoard and W&B |
| T-099 | [ ] Integrate experiment tracking into ML estimator training script | P1 | T-095, T-066 | 1h | Training logs to both TensorBoard and W&B |
| T-100 | [ ] Integrate experiment tracking into benchmark runner | P1 | T-095, T-086 | 1h | Benchmark results logged to both backends |

---

## Phase 11: Paper Infrastructure

| ID | Task | Priority | Dependencies | Effort | Acceptance Criteria |
|----|------|----------|-------------|--------|-------------------|
| T-101 | [ ] Set up `paper/figures/` directory structure | P2 | None | 15min | All subdirectories exist (architecture, performance, comparison, environment) |
| T-102 | [ ] Set up `paper/tables/` directory structure | P2 | None | 15min | Directory exists for CSV/JSON table data |
| T-103 | [ ] Create `paper/references.bib` — Bibliography file | P2 | None | 1h | BibTeX entries for YOLO, tracking, Roboflow, density estimation papers |
| T-104 | [ ] Generate architecture diagram (Figure 1) | P2 | T-087 | 1h | SVG diagram of system architecture saved to `paper/figures/architecture/` |
| T-105 | [ ] Generate pipeline data flow diagram (Figure 2) | P2 | T-087 | 1h | SVG diagram of data flow saved to `paper/figures/architecture/` |
| T-106 | [ ] Generate FPS comparison chart (Figure 3) | P2 | T-088, T-087 | 30min | Bar chart saved to `paper/figures/performance/` |
| T-107 | [ ] Generate latency distribution box plots (Figure 4) | P2 | T-089, T-087 | 30min | Box plots saved to `paper/figures/performance/` |
| T-108 | [ ] Generate MAE comparison chart (Figure 5) | P2 | T-090, T-087 | 30min | Bar chart saved to `paper/figures/performance/` |
| T-109 | [ ] Generate actual vs predicted density time series (Figure 6) | P2 | T-091, T-087 | 30min | Time series plot saved to `paper/figures/performance/` |
| T-110 | [ ] Generate RMSE/MAPE across LSTM/GRU/TCN chart (Figure 7) | P2 | T-091, T-087 | 30min | Grouped bar chart saved to `paper/figures/comparison/` |
| T-111 | [ ] Generate environment comparison chart (Figure 8) | P2 | T-092, T-087 | 30min | Grouped bar chart saved to `paper/figures/environment/` |
| T-112 | [ ] Generate optimization impact chart (Figure 9) | P2 | T-093, T-087 | 30min | Before/after bar chart saved to `paper/figures/performance/` |
| T-113 | [ ] Export all table data (Tables 1-6) | P2 | T-088-T-093 | 1h | CSV/JSON files for all 6 tables saved to `paper/tables/` |
| T-114 | [ ] Create `scripts/07_generate_paper_figures.py` — Figure generation entry point | P2 | T-104-T-113 | 1h | CLI tool to regenerate all figures from results data |

---

## Phase 12: Vision Transformer Segmentation (Future Optimization)

| ID | Task | Priority | Dependencies | Effort | Acceptance Criteria |
|----|------|----------|-------------|--------|-------------------|
| T-115 | [ ] Research and select ViT segmentation architecture | P3 | None | 2h | Architecture decision documented (ViT + Mask2Former or SETR) |
| T-116 | [ ] Implement `src/models/segmentation/vit_seg.py` — ViT segmentation model | P3 | T-115 | 4h | ViT-based segmentation model with configurable backbone and decoder |
| T-117 | [ ] Train ViT segmentation model | P3 | T-116, T-027 | 6-12h | Model trained, checkpoint saved, metrics logged |
| T-118 | [ ] Benchmark ViT segmentation vs YOLO detection | P3 | T-117, T-086 | 1h | Results saved, accuracy/speed comparison available |
| T-119 | [ ] Write unit tests for ViT segmentation model | P3 | T-116, T-014 | 1h | Tests forward pass, mask output shapes |

---

## Phase 13: Documentation & Final Review

| ID | Task | Priority | Dependencies | Effort | Acceptance Criteria |
|----|------|----------|-------------|--------|-------------------|
| T-120 | [ ] Write comprehensive README.md | P1 | All phases complete | 2h | Project overview, setup instructions, usage examples, architecture description |
| T-121 | [ ] Add docstrings to all public classes and functions | P1 | All phases complete | 4h | Every public API has docstring with args, returns, examples |
| T-122 | [ ] Run full test suite and fix any failures | P0 | All phases complete | 2h | `pytest tests/ -v` passes with 100% pass rate |
| T-123 | [ ] Run linting and fix style issues | P2 | All phases complete | 1h | No linting errors |
| T-124 | [ ] Verify all experiment configs are valid and runnable | P1 | All phases complete | 1h | Each config file loads without errors |
| T-125 | [ ] Verify end-to-end pipeline on test video | P0 | All phases complete | 1h | Full pipeline (detect → track → filter → predict) runs on test video |
| T-126 | [ ] Verify all paper figures are generated correctly | P2 | T-114 | 30min | All 9 figures exist and are publication-quality |
| T-127 | [ ] Cross-reference TASKS.md with SPEC.md for completeness | P1 | All phases complete | 1h | Every spec section has corresponding tasks |
| T-128 | [ ] Final review of PROBLEM.md for accuracy and completeness | P1 | All phases complete | 30min | Problem statement reflects actual implementation |

---

## Execution Order Summary

```
Phase 0  ──▶ Phase 1  ──▶ Phase 2 ──▶ Phase 3 ──▶ Phase 4 ──▶ Phase 5 ──▶ Phase 6 ──▶ Phase 7
                                                                                         │
                                                                                         ▼
Phase 10 (parallel)                                              Phase 8 ──▶ Phase 9
                                                                                         │
                                                                                         ▼
Phase 13 ◀── Phase 12 ◀── Phase 11 ◀───────────────────────────── Phase 9
```

**Parallel tracks:**

- Phase 10 (Experiment Tracking) can begin as soon as Phase 0 is complete
- Phase 12 (ViT) can begin after Phase 2, runs parallel to Phases 3-9
- Phase 11 (Paper) begins after Phase 9 results are available

---

## Task Statistics

| Category | Count |
|----------|-------|
| Total tasks | 128 |
| P0 (Critical) | 32 |
| P1 (High) | 48 |
| P2 (Medium) | 40 |
| P3 (Low/Future) | 8 |

| Phase | Tasks | Estimated Total Effort |
|-------|-------|----------------------|
| Phase 0: Scaffolding | 15 | ~12h |
| Phase 1: Data Pipeline | 15 | ~22h |
| Phase 2: Detection Training | 14 | ~25h (mostly GPU time) |
| Phase 3: Tracking | 5 | ~6h |
| Phase 4: Spatial Logic | 7 | ~9h |
| Phase 5: Feature Engineering | 5 | ~6h |
| Phase 6: ML Estimator | 9 | ~18h (mostly GPU time) |
| Phase 7: Inference Pipeline | 7 | ~14h |
| Phase 8: Optimization | 7 | ~8h |
| Phase 9: Evaluation | 10 | ~14h |
| Phase 10: Experiment Tracking | 6 | ~6h |
| Phase 11: Paper Infrastructure | 14 | ~7h |
| Phase 12: ViT Segmentation (Future) | 5 | ~18h |
| Phase 13: Documentation | 9 | ~12h |
