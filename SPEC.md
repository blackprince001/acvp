# Technical Specification: Real-Time Traffic Density Estimation

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Project Structure](#2-project-structure)
3. [Data Layer](#3-data-layer)
4. [CV Model Training Pipeline](#4-cv-model-training-pipeline)
5. [Tracking Integration](#5-tracking-integration)
6. [Spatial Logic Layer](#6-spatial-logic-layer)
7. [Feature Engineering](#7-feature-engineering)
8. [ML Estimator](#8-ml-estimator)
9. [Inference Pipeline](#9-inference-pipeline)
10. [Model Optimization](#10-model-optimization)
11. [Evaluation & Benchmarking](#11-evaluation--benchmarking)
12. [Experiment Tracking](#12-experiment-tracking)
13. [Configuration System](#13-configuration-system)
14. [Testing Strategy](#14-testing-strategy)
15. [Dependencies](#15-dependencies)
16. [Future Work](#16-future-work)

---

## 1. System Architecture

### 1.1 High-Level Design

The system implements a **two-stage pipeline** architecture:

```
┌──────────────────────────────────────────────────────────────────┐
│                        STAGE 1: CV PIPELINE                      │
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────────────┐   │
│  │  YOLO    │───▶│ Tracking │───▶│   Spatial Logic Layer    │   │
│  │  (Det/   │    │ (BoT-    │    │   (Road Mask + IoU       │   │
│  │  Seg)    │    │  SORT/   │    │    Filtering)            │   │
│  │          │    │  Byte)   │    │                          │   │
│  └──────────┘    └──────────┘    └────────────┬─────────────┘   │
│                                                │                 │
│                                    ┌───────────▼──────────┐     │
│                                    │   Telemetry Stream   │     │
│                                    │   (aggregated per    │     │
│                                    │    N-frame window)   │     │
│                                    │   - Vehicle counts   │     │
│                                    │   - Occupancy ratios │     │
│                                    │   - Movement vectors │     │
│                                    │   - Temporal indices │     │
│                                    └──────────────────────┘     │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                     STAGE 2: ML ESTIMATOR                        │
│                                                                  │
│  ┌──────────────────┐    ┌──────────────────────────────────┐   │
│  │  Feature Window  │───▶│  LSTM / GRU / TCN                │   │
│  │  (T-N ... T)     │    │  (PyTorch native)                │   │
│  └──────────────────┘    └────────────┬─────────────────────┘   │
│                                       │                          │
│                          ┌────────────▼─────────────────────┐   │
│                          │  Density Prediction (T+1 ... T+N)│   │
│                          │  - Predicted vehicle counts      │   │
│                          │  - Predicted occupancy ratios    │   │
│                          │  - Confidence intervals          │   │
│                          └──────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 Component Responsibilities

| Component | Responsibility | Output |
|-----------|---------------|--------|
| YOLO Detection | Bounding box detection of vehicles and roads | Boxes, class IDs, confidence scores |
| YOLO Segmentation | Instance segmentation of vehicles and roads | Masks, class IDs, confidence scores |
| Tracking | Persistent ID assignment across frames | Tracked objects with IDs, trajectories |
| Spatial Logic | Filter on-road vs off-road vehicles | Filtered vehicle list, occupancy data |
| Feature Engineering | Compute density features per N-frame window | Feature vectors per window |
| ML Estimator | Predict future density states | N-step ahead predictions |
| Inference Orchestrator | Coordinate all components at fixed FPS | Unified density stream |
| Evaluation | Benchmark performance across configurations | Metrics, graphs, tables |

### 1.3 Data Flow

```
Raw Video ──▶ Preprocessed Frames ──▶ YOLO Inference ──▶ Tracking
                                                          │
                                                          ▼
Road Masks ──▶ Spatial Logic ──▶ Filtered Vehicles ──▶ Feature Engineering
               (per N-frame window)                          │
                                                             ▼
                                                     ML Estimator Input
                                                             │
                                                             ▼
                                                     Density Predictions
```

### 1.4 Execution Modes

| Mode | Components Active | Use Case |
|------|------------------|----------|
| `detect` | YOLO Detection + Tracking | Baseline, edge deployment |
| `segment` | YOLO Segmentation + Tracking + Spatial Logic | High-precision mode |
| `predict` | Full Stage 1 + ML Estimator | Predictive mode |
| `benchmark` | All components + Evaluation | Experimentation |

---

## 2. Project Structure

```
aai-computer-vision-project/
├── PROBLEM.md                          # Problem statement (this document's companion)
├── SPEC.md                             # This specification
├── TASKS.md                            # Sequential task list
│
├── configs/                            # YAML configuration files
│   ├── base.yaml                       # Base configuration (defaults)
│   ├── models/
│   │   ├── detection/
│   │   │   ├── yolov8n.yaml
│   │   │   ├── yolov8s.yaml
│   │   │   ├── yolov8m.yaml
│   │   │   ├── yolov8l.yaml
│   │   │   ├── yolov11n.yaml
│   │   │   ├── yolov11s.yaml
│   │   │   ├── yolov11m.yaml
│   │   │   └── yolov11l.yaml
│   │   ├── segmentation/
│   │   │   ├── yolov8n-seg.yaml
│   │   │   ├── yolov8s-seg.yaml
│   │   │   ├── yolov11n-seg.yaml
│   │   │   └── yolov11s-seg.yaml
│   │   └── ml_estimator/
│   │       ├── lstm.yaml
│   │       ├── gru.yaml
│   │       └── tcn.yaml
│   ├── tracking/
│   │   ├── botsort.yaml
│   │   └── bytetrack.yaml
│   └── experiments/
│       ├── baseline_detection.yaml
│       ├── baseline_segmentation.yaml
│       ├── predictive_lstm.yaml
│       ├── predictive_gru.yaml
│       ├── predictive_tcn.yaml
│       └── edge_deployment.yaml
│
├── data/
│   ├── raw/                            # Original video files / image sequences
│   │   ├── <scene_name>/
│   │   │   ├── video.mp4               # Or individual frames
│   │   │   └── metadata.yaml           # Scene metadata (location, type, camera info)
│   ├── processed/                      # Preprocessed frames
│   │   ├── <scene_name>/
│   │   │   ├── frames/                 # Resized, normalized frames
│   │   │   └── metadata.yaml
│   ├── segmented/                      # Roboflow-segmented masks
│   │   ├── <scene_name>/
│   │   │   ├── masks/                  # PNG mask files
│   │   │   ├── annotations.yaml        # Multi-label annotations
│   │   │   └── road_masks/             # Static road boundary masks
│   ├── splits/                         # Train/val/test split manifests
│   │   ├── train.txt
│   │   ├── val.txt
│   │   ├── test.txt
│   │   └── split_config.yaml
│   └── generated/                      # CV pipeline output streams
│       ├── <experiment_name>/
│       │   ├── telemetry.csv           # Time-series telemetry data
│       │   └── features.parquet        # Engineered features
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/                           # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── raw_loader.py               # Video/frame loading utilities
│   │   ├── preprocess.py               # Frame preprocessing pipeline
│   │   ├── roboflow_segmentation.py    # Roboflow API segmentation pipeline
│   │   ├── dataset.py                  # PyTorch Dataset implementations
│   │   ├── augmentations.py            # Training-time augmentations
│   │   └── splits.py                   # Train/val/test split management
│   │
│   ├── models/                         # Model definitions
│   │   ├── __init__.py
│   │   ├── detection/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                 # Base detection model interface
│   │   │   ├── yolov8.py              # YOLOv8 detection wrapper
│   │   │   ├── yolov11.py             # YOLO11 detection wrapper
│   │   │   └── registry.py            # Model registry for config-based loading
│   │   ├── segmentation/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                 # Base segmentation model interface
│   │   │   ├── yolov8_seg.py          # YOLOv8 segmentation wrapper
│   │   │   └── yolov11_seg.py         # YOLO11 segmentation wrapper
│   │   └── ml_estimator/
│   │       ├── __init__.py
│   │       ├── base.py                 # Base estimator interface
│   │       ├── lstm.py                 # LSTM architecture
│   │       ├── gru.py                  # GRU architecture
│   │       └── tcn.py                  # Temporal Convolutional Network
│   │
│   ├── tracking/                       # Object tracking
│   │   ├── __init__.py
│   │   ├── base.py                     # Base tracker interface
│   │   └── ultralytics_tracker.py      # Ultralytics built-in tracking wrapper
│   │
│   ├── spatial_logic/                  # Spatial Logic Layer
│   │   ├── __init__.py
│   │   ├── road_mask.py                # Road mask generation and management
│   │   ├── intersection.py             # Mask overlap computation (IoU-based)
│   │   ├── filtering.py                # On-road classification logic
│   │   └── occupancy.py                # Pixel occupancy ratio computation
│   │
│   ├── features/                       # Feature engineering
│   │   ├── __init__.py
│   │   ├── extractor.py                # Feature extraction from telemetry
│   │   ├── windowing.py                # Temporal windowing utilities
│   │   └── normalizer.py               # Feature normalization
│   │
│   ├── inference/                      # Inference pipeline
│   │   ├── __init__.py
│   │   ├── orchestrator.py             # Main pipeline orchestrator
│   │   ├── queue.py                    # Inter-component communication queues
│   │   └── video_io.py                 # Video input/output handling
│   │
│   ├── optimization/                   # Model optimization
│   │   ├── __init__.py
│   │   ├── onnx_export.py              # ONNX export utilities
│   │   └── benchmark.py                # Optimization benchmarking
│   │
│   ├── evaluation/                     # Evaluation and metrics
│   │   ├── __init__.py
│   │   ├── metrics.py                  # Metric implementations
│   │   ├── benchmark.py                # Benchmark runner
│   │   └── visualization.py            # Graph and chart generation
│   │
│   ├── experiment/                     # Experiment tracking
│   │   ├── __init__.py
│   │   ├── tracker.py                  # Unified experiment tracker interface
│   │   ├── tensorboard_logger.py       # TensorBoard integration
│   │   └── wandb_logger.py             # Weights & Biases integration
│   │
│   └── utils/                          # Shared utilities
│       ├── __init__.py
│       ├── config.py                   # Configuration loading and validation
│       ├── logging.py                  # Logging setup
│       ├── visualization.py            # General visualization utilities
│       └── types.py                    # Shared type definitions
│
├── scripts/                            # Executable pipeline scripts
│   ├── 01_preprocess_data.py
│   ├── 02_run_roboflow_segmentation.py
│   ├── 03_train_detection.py
│   ├── 04_train_segmentation.py
│   ├── 05_train_ml_estimator.py
│   ├── 06_run_inference.py
│   ├── 07_evaluate.py
│   └── 08_export_models.py
│
├── experiments/                        # Experiment artifacts
│   ├── logs/                           # Training logs
│   │   ├── tensorboard/
│   │   └── wandb/
│   ├── checkpoints/                    # Saved model weights
│   │   ├── detection/
│   │   ├── segmentation/
│   │   └── ml_estimator/
│   └── results/                        # Evaluation results
│       ├── benchmarks/
│       ├── metrics/
│       └── predictions/
│
├── tests/                              # Test suite
│   ├── conftest.py
│   ├── test_data/
│   │   ├── test_raw_loader.py
│   │   ├── test_preprocess.py
│   │   └── test_dataset.py
│   ├── test_models/
│   │   ├── test_detection.py
│   │   ├── test_segmentation.py
│   │   └── test_ml_estimator.py
│   ├── test_tracking/
│   │   └── test_tracker.py
│   ├── test_spatial_logic/
│   │   ├── test_road_mask.py
│   │   ├── test_intersection.py
│   │   └── test_filtering.py
│   ├── test_features/
│   │   └── test_extractor.py
│   ├── test_inference/
│   │   └── test_orchestrator.py
│   └── test_evaluation/
│       └── test_metrics.py
│
├── pyproject.toml                      # Project dependencies and metadata
├── main.py                             # Entry point
└── .gitignore
```

---

## 3. Data Layer

### 3.1 Raw Data Ingestion

**Input:** User-provided video files or image sequences stored in `data/raw/<scene_name>/`.
**Metadata Schema** (`data/raw/<scene_name>/metadata.yaml`):

```yaml
scene:
  name: "highway_downtown"
  type: "highway"          # highway | urban | suburban | intersection
  location: "City, Country"
  camera:
    height_m: 8.5
    angle_deg: 35
    resolution: [1920, 1080]
    fps: 30
  recording:
    date: "2024-06-15"
    weather: "clear"       # clear | rain | fog | night
    duration_seconds: 3600
  road:
    lanes: 4
    has_sidewalk: true
    has_service_road: false
```

**RawLoader** (`src/data/raw_loader.py`):

- Supports MP4, AVI, MKV video formats
- Supports image sequences (JPEG, PNG)
- Extracts frames at configurable intervals
- Returns frame iterator with metadata

### 3.2 Preprocessing Pipeline

**Output:** `data/processed/<scene_name>/frames/`
**Steps:**

1. **Frame extraction** — Extract frames at target FPS (configurable, default: original FPS)
2. **Resizing** — Resize to model input dimensions (640x640 for YOLO, configurable)
3. **Normalization** — Standard ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
4. **Frame indexing** — Create index file mapping frame number to file path
**Preprocessor** (`src/data/preprocess.py`):

```python
class FramePreprocessor:
    def __init__(self, target_size: tuple[int, int] = (640, 640),
                 target_fps: int | None = None,
                 normalize: bool = True):
        ...
    def process(self, video_path: str, output_dir: str) -> list[str]:
        """Extract and preprocess frames from video."""
        ...
```

### 3.3 Roboflow Segmentation Pipeline

**Purpose:** Send preprocessed frames to a Roboflow endpoint for automated segmentation of roads and vehicles. Segmentation runs remotely via the Roboflow API — nothing runs locally.
**Output:** `data/segmented/<scene_name>/`
**Classes to Segment:**

| Class ID | Class Name | Type |
|----------|-----------|------|
| 0 | road | static |
| 1 | vehicle | dynamic |
The vehicle class covers all vehicle subtypes (cars, buses, trucks, motorcycles, vans, etc.). No fine-grained vehicle classification is performed — the goal is to detect all road-bound vehicles for density counting.
**Roboflow Pipeline** (`src/data/roboflow_segmentation.py`):

```python
class RoboflowSegmentationPipeline:
    def __init__(self, api_key: str, workspace: str, project: str,
                 model_version: str = "latest"):
        ...
    def segment_scene(self, frames_dir: str, output_dir: str,
                      batch_size: int = 32) -> dict:
        """
        Send frames to Roboflow API for segmentation.
        Downloads and organizes returned masks.
        """
        ...
    def generate_road_mask(self, masks: np.ndarray,
                           class_ids: np.ndarray) -> np.ndarray:
        """Extract and merge road region masks across frames."""
        ...
```

**Workflow:**

1. Load preprocessed frames
2. Batch frames and send to Roboflow inference endpoint
3. Download returned segmentation masks (road + vehicle)
4. Generate per-frame mask files (PNG, single-channel, class ID encoded)
5. Generate static road mask (merged across frames for consistency)
6. Save annotations in YAML format with class distributions

### 3.4 Dataset Classes

**PyTorch Dataset** (`src/data/dataset.py`):

```python
class TrafficDetectionDataset(Dataset):
    """Dataset for YOLO detection training."""
    def __init__(self, frames_dir: str, annotations_dir: str,
                 transforms: Callable | None = None):
        ...
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        """Returns (image, {boxes, labels, image_id})."""
        ...
class TrafficSegmentationDataset(Dataset):
    """Dataset for YOLO segmentation training."""
    def __init__(self, frames_dir: str, masks_dir: str,
                 annotations_dir: str, transforms: Callable | None = None):
        ...
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        """Returns (image, {boxes, labels, masks, image_id})."""
        ...
class TelemetryDataset(Dataset):
    """Dataset for ML Estimator training."""
    def __init__(self, telemetry_path: str, window_size: int,
                 prediction_horizon: int):
        ...
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (input_window, target_window)."""
        ...
```

### 3.5 Data Augmentations

**Training-time augmentations** (`src/data/augmentations.py`):

- Random brightness/contrast adjustment (lighting variation)
- Random Gaussian blur (camera quality variation)
- Random rotation (±5 degrees, camera angle variation)
- Random horizontal flip (for symmetric road views)
All augmentations are random/stochastic. No mosaic, MixUp, or structured augmentations.
**Note:** Augmentations are applied during training only. Inference uses raw preprocessed frames.

### 3.6 Train/Val/Test Splits

**Split Strategy** (`src/data/splits.py`):

- **By scene:** 70% train, 15% val, 15% test (scene-level split, not frame-level)
- Ensures no data leakage between splits
- Stratified by scene type (highway, urban, etc.)
- Split manifests saved to `data/splits/`

---

## 4. CV Model Training Pipeline

### 4.1 Detection Models

**Models to Train:**

| Model | Size | Parameters | Target FPS |
|-------|------|-----------|------------|
| YOLOv8n | 640 | 3.2M | ~200 |
| YOLOv8s | 640 | 11.2M | ~120 |
| YOLOv8m | 640 | 25.9M | ~70 |
| YOLOv8l | 640 | 43.7M | ~45 |
| YOLO11n | 640 | 2.6M | ~220 |
| YOLO11s | 640 | 9.4M | ~130 |
| YOLO11m | 640 | 20.1M | ~80 |
| YOLO11l | 640 | 25.3M | ~55 |
**Training Configuration:**

```yaml
# configs/models/detection/yolov8s.yaml (example)
model:
  name: "yolov8s"
  task: "detect"
  pretrained: true
  pretrained_weights: "yolov8s.pt"
training:
  epochs: 100
  batch_size: 16
  image_size: 640
  optimizer: "AdamW"
  lr: 0.001
  lr_scheduler: "cosine"
  warmup_epochs: 5
  weight_decay: 0.0005
  patience: 20  # early stopping patience
  augmentations:
    brightness: 0.3
    contrast: 0.3
    blur: 0.1
    rotation: 5
    flip_lr: 0.5
  classes:
    - vehicle
    - road
```

**Detection Model Wrapper** (`src/models/detection/yolov8.py`):

```python
class YOLOv8Detector:
    def __init__(self, model_size: str = "s", pretrained: bool = True,
                 device: str = "cuda"):
        ...
    def train(self, data_config: dict, epochs: int = 100, **kwargs):
        ...
    def predict(self, image: np.ndarray, conf_threshold: float = 0.25) -> dict:
        """Returns {boxes, confidences, class_ids}."""
        ...
    def export_onnx(self, output_path: str, opset: int = 17):
        ...
```

### 4.2 Segmentation Models

**Purpose:** YOLO segmentation models produce pixel-exact masks for vehicles and roads. This enables the Spatial Logic Layer to compute precise mask-road overlap, filtering out parked cars, sidewalk vehicles, and other off-road noise that bounding boxes would incorrectly include. This is the high-accuracy density estimation path.
**Models to Train:**

| Model | Size | Parameters | Target FPS |
|-------|------|-----------|------------|
| YOLOv8n-seg | 640 | 3.4M | ~150 |
| YOLOv8s-seg | 640 | 11.8M | ~90 |
| YOLO11n-seg | 640 | 2.8M | ~160 |
| YOLO11s-seg | 640 | 10.1M | ~95 |
**Training Configuration:** Same structure as detection, with `task: "segment"`.
**Segmentation Model Wrapper** (`src/models/segmentation/yolov8_seg.py`):

```python
class YOLOv8Segmenter:
    def __init__(self, model_size: str = "s", pretrained: bool = True,
                 device: str = "cuda"):
        ...
    def train(self, data_config: dict, epochs: int = 100, **kwargs):
        ...
    def predict(self, image: np.ndarray, conf_threshold: float = 0.25) -> dict:
        """Returns {boxes, confidences, class_ids, masks}."""
        ...
```

### 4.3 Training Infrastructure

**Training Script** (`scripts/03_train_detection.py`, `scripts/04_train_segmentation.py`):

- Loads config from YAML
- Initializes dataset with augmentations
- Creates model from registry
- Sets up experiment tracking (TensorBoard + W&B)
- Runs training loop with checkpointing
- Saves best and last checkpoints
- Generates training curves
**Checkpointing Strategy:**
- Save every epoch (last checkpoint)
- Save when validation mAP improves (best checkpoint)
- Save with metadata: epoch, metrics, config hash
**Experiment Tracking Integration:**
- Log per-epoch: loss, mAP@50, mAP@50:95, precision, recall
- Log per-batch: batch loss, learning rate
- Log images: prediction overlays on validation set every 10 epochs
- Log hyperparameters: full config dump

---

## 5. Tracking Integration

### 5.1 Tracker Selection

**Ultralytics Built-in Tracking:**

- BoT-SORT: Better for crowded scenes, handles occlusion well
- ByteTrack: Faster, good for moderate traffic density
**Tracker Wrapper** (`src/tracking/ultralytics_tracker.py`):

```python
class UltralyticsTracker:
    def __init__(self, tracker_type: str = "botsort",  # botsort | bytetrack
                 track_high_thresh: float = 0.5,
                 track_low_thresh: float = 0.1,
                 new_track_thresh: float = 0.6,
                 track_buffer: int = 30,
                 match_thresh: float = 0.8,
                 fuse_score: bool = True):
        ...
    def update(self, frame: np.ndarray) -> list[dict]:
        """
        Returns list of tracked objects:
        [{id, box, class_id, confidence, mask (if seg)}]
        """
        ...
    def get_trajectories(self, min_length: int = 5) -> dict[int, list[tuple]]:
        """Returns persistent trajectories for objects tracked >= min_length frames."""
        ...
```

### 5.2 Movement Vector Calculation

From tracked objects with persistent IDs:

```python
def compute_movement_vectors(tracks_current: list[dict],
                             tracks_previous: list[dict],
                             fps: float,
                             pixel_to_meter_ratio: float) -> list[dict]:
    """
    Computes for each tracked object:
    - velocity (pixels/frame and m/s)
    - direction (angle in degrees)
    - acceleration (if 3+ frames available)
    """
    ...
```

**Output per tracked object:**

```python
{
    "id": 42,
    "box": [x1, y1, x2, y2],
    "class_id": 0,
    "confidence": 0.92,
    "mask": np.ndarray,  # if segmentation mode
    "centroid": (cx, cy),
    "velocity": {"x": 2.3, "y": -0.5, "magnitude": 2.35},
    "direction_deg": 347.8,
    "frames_tracked": 15,
}
```

### 5.3 Redundant Counting Prevention

**Strategy:**

- Each vehicle ID is counted once per zone crossing
- Zone-based counting: define entry/exit lines or regions
- For density estimation (snapshot), count all active on-road IDs per frame
- For flow estimation (vehicles/time), count unique IDs crossing zone boundaries

---

## 6. Spatial Logic Layer

### 6.1 Road Mask Management

**Road Mask** (`src/spatial_logic/road_mask.py`):
The road mask is a binary image where `1` represents road pixels and `0` represents non-road pixels.

```python
class RoadMaskManager:
    def __init__(self):
        self.static_mask: np.ndarray | None = None  # Pre-computed road mask
        self.dynamic_mask: np.ndarray | None = None  # For construction zones (future)
    def load_static_mask(self, mask_path: str) -> np.ndarray:
        """Load pre-computed road mask from segmentation output."""
        ...
    def generate_from_segmentation(self, seg_masks: np.ndarray,
                                   class_ids: np.ndarray) -> np.ndarray:
        """Merge road-class masks across frames to create static road mask."""
        ...
    def get_mask(self, frame_idx: int | None = None) -> np.ndarray:
        """Return road mask (static or dynamic based on frame)."""
        ...
```

**Static Road Mask Generation:**

1. From Roboflow segmentation output, extract all frames' road masks
2. Compute pixel-wise mode (most common class per pixel across frames)
3. Apply morphological operations to clean up noise
4. Save as single-channel PNG

### 6.2 Intersection Algorithm

**IoU-Based Filtering** (`src/spatial_logic/intersection.py`):

```python
def compute_mask_road_overlap(vehicle_mask: np.ndarray,
                              road_mask: np.ndarray) -> float:
    """
    Compute what fraction of the vehicle mask overlaps with the road mask.
    Returns: overlap_ratio in [0, 1]
    - 1.0 = vehicle entirely on road
    - 0.0 = vehicle entirely off road
    """
    intersection = np.logical_and(vehicle_mask, road_mask).sum()
    vehicle_area = vehicle_mask.sum()
    return intersection / vehicle_area if vehicle_area > 0 else 0.0
def compute_box_road_overlap(vehicle_box: np.ndarray,
                             road_mask: np.ndarray) -> float:
    """
    Compute overlap using bounding box (faster, less accurate).
    Used in detection-only mode when masks are unavailable.
    """
    x1, y1, x2, y2 = vehicle_box
    roi = road_mask[y1:y2, x1:x2]
    road_pixels = roi.sum()
    total_pixels = roi.size
    return road_pixels / total_pixels if total_pixels > 0 else 0.0
```

### 6.3 On-Road Classification

**Filtering Logic** (`src/spatial_logic/filtering.py`):

```python
class OnRoadFilter:
    def __init__(self, overlap_threshold: float = 0.5,
                 min_velocity: float = 0.5,  # m/s, to filter parked cars
                 use_velocity_filter: bool = True):
        self.overlap_threshold = overlap_threshold
        self.min_velocity = min_velocity
        self.use_velocity_filter = use_velocity_filter
    def classify(self, tracked_objects: list[dict],
                 road_mask: np.ndarray) -> list[dict]:
        """
        Classify each tracked object as on-road or off-road.
        Criteria:
        1. Mask/box overlap with road > threshold
        2. (Optional) Velocity > min_velocity (filters parked cars)
        3. Not in excluded zones (parking lots, service roads)
        """
        ...
    def filter_on_road(self, tracked_objects: list[dict],
                       road_mask: np.ndarray) -> list[dict]:
        """Return only on-road vehicles."""
        classified = self.classify(tracked_objects, road_mask)
        return [obj for obj in classified if obj.get("is_on_road", False)]
```

**Classification Criteria:**

| Criterion | Detection Mode | Segmentation Mode |
|-----------|---------------|-------------------|
| Overlap method | Box-road IoU | Mask-road IoU |
| Threshold | 0.6 | 0.5 |
| Velocity filter | Optional | Optional |
| Accuracy | Medium | High |

### 6.4 Occupancy Ratio Computation

**Pixel Occupancy** (`src/spatial_logic/occupancy.py`):

```python
def compute_occupancy_ratio(vehicle_masks: list[np.ndarray],
                            road_mask: np.ndarray) -> float:
    """
    Compute the fraction of road pixels occupied by vehicles.
    occupancy = (road_pixels covered by vehicles) / (total road pixels)
    """
    combined_vehicle_mask = np.zeros_like(road_mask, dtype=bool)
    for mask in vehicle_masks:
        combined_vehicle_mask = np.logical_or(combined_vehicle_mask, mask)
    occupied_road = np.logical_and(combined_vehicle_mask, road_mask).sum()
    total_road = road_mask.sum()
    return occupied_road / total_road if total_road > 0 else 0.0
```

---

## 7. Feature Engineering

### 7.1 Feature Extraction

**Features** (`src/features/extractor.py`):
Density estimation is computed **per N-frame window**, not per individual frame. Features are aggregated over the window to produce a single feature vector per window.

| Feature | Description | Type |
|---------|------------|------|
| `vehicle_count` | Total on-road vehicles (mean over window) | Scalar |
| `vehicle_count_std` | Vehicle count std deviation over window | Scalar |
| `occupancy_ratio` | Road pixel occupancy (mean over window) | Scalar [0, 1] |
| `occupancy_ratio_std` | Occupancy std deviation over window | Scalar |
| `avg_velocity` | Mean velocity of all vehicles (mean over window) | Scalar (m/s) |
| `std_velocity` | Velocity std deviation (mean over window) | Scalar |
| `avg_direction` | Mean movement direction (mean over window) | Scalar (degrees) |
| `density_index` | Normalized density (count / road_area) | Scalar |
| `flow_rate` | Vehicles crossing zone per second | Scalar |
| `congestion_level` | Discrete level (0-4) derived from occupancy | Categorical |

```python
class FeatureExtractor:
    def __init__(self, road_mask: np.ndarray, pixel_to_meter: float,
                 window_size: int = 30):
        ...
    def extract_window(self, frames_data: list[dict]) -> dict:
        """Extract aggregated features for an N-frame window."""
        ...
    def extract_batch(self, all_frames_data: list[list[dict]]) -> pd.DataFrame:
        """Extract features for multiple windows, return DataFrame."""
        ...
```

### 7.2 Temporal Windowing

**Windowing** (`src/features/windowing.py`):

```python
class TemporalWindowBuilder:
    def __init__(self, window_size: int = 10,  # number of feature windows
                 prediction_horizon: int = 5,   # windows to predict
                 stride: int = 1):
        ...
    def build_windows(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Build sliding windows for sequence model training.
        Input:  features[T-window_size : T]
        Output: features[T+1 : T+prediction_horizon]
        """
        ...
```

### 7.3 Feature Normalization

**Normalization** (`src/features/normalizer.py`):

```python
class FeatureNormalizer:
    def __init__(self, method: str = "standard"):  # standard | minmax | robust
        ...
    def fit(self, features: pd.DataFrame):
        """Fit normalizer on training data."""
        ...
    def transform(self, features: pd.DataFrame) -> np.ndarray:
        """Transform features to normalized array."""
        ...
    def fit_transform(self, features: pd.DataFrame) -> np.ndarray:
        ...
    def inverse_transform(self, normalized: np.ndarray) -> np.ndarray:
        """Inverse transform predictions back to original scale."""
        ...
```

---

## 8. ML Estimator

### 8.1 Architecture Overview

The ML Estimator is a sequence model that takes a window of historical density features and predicts future density states.
**Input:** Feature window of shape `(window_size, num_features)`
**Output:** Predictions of shape `(prediction_horizon, num_targets)`

### 8.2 LSTM Architecture

**Configuration** (`configs/models/ml_estimator/lstm.yaml`):

```yaml
model:
  name: "lstm"
  input_size: 10          # num_features
  hidden_size: 128
  num_layers: 2
  dropout: 0.2
  bidirectional: false
training:
  epochs: 200
  batch_size: 64
  optimizer: "Adam"
  lr: 0.001
  lr_scheduler: "reduce_on_plateau"
  patience: 30
  loss: "mse"
  clip_grad_norm: 1.0
```

**Implementation** (`src/models/ml_estimator/lstm.py`):

```python
class LSTMEstimator(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int, output_size: int,
                 dropout: float = 0.2, bidirectional: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        lstm_out_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Use last time step output
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions
```

### 8.3 GRU Architecture

**Configuration** (`configs/models/ml_estimator/gru.yaml`):

```yaml
model:
  name: "gru"
  input_size: 10
  hidden_size: 128
  num_layers: 2
  dropout: 0.2
  bidirectional: false
training:
  epochs: 200
  batch_size: 64
  optimizer: "Adam"
  lr: 0.001
  lr_scheduler: "reduce_on_plateau"
  patience: 30
  loss: "mse"
  clip_grad_norm: 1.0
```

**Implementation** (`src/models/ml_estimator/gru.py`):

```python
class GRUEstimator(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int, output_size: int,
                 dropout: float = 0.2, bidirectional: bool = False):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        gru_out_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Sequential(
            nn.Linear(gru_out_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        predictions = self.fc(gru_out[:, -1, :])
        return predictions
```

### 8.4 TCN Architecture

**Configuration** (`configs/models/ml_estimator/tcn.yaml`):

```yaml
model:
  name: "tcn"
  input_size: 10
  num_channels: [64, 64, 64, 64]
  kernel_size: 3
  dropout: 0.2
  dilation_base: 2
training:
  epochs: 200
  batch_size: 64
  optimizer: "Adam"
  lr: 0.001
  lr_scheduler: "reduce_on_plateau"
  patience: 30
  loss: "mse"
  clip_grad_norm: 1.0
```

**Implementation** (`src/models/ml_estimator/tcn.py`):

```python
class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
class TCNEstimator(nn.Module):
    def __init__(self, input_size: int, num_channels: list[int],
                 kernel_size: int, output_size: int, dropout: float = 0.2,
                 dilation_base: int = 2):
        super().__init__()
        layers = []
        for i, channels in enumerate(num_channels):
            dilation = dilation_base ** i
            layers.append(TemporalBlock(
                input_size if i == 0 else num_channels[i-1],
                channels, kernel_size, dilation, dropout
            ))
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size) -> (batch, input_size, seq_len)
        x = x.transpose(1, 2)
        out = self.tcn(x)
        # Use last time step
        predictions = self.fc(out[:, :, -1])
        return predictions
```

### 8.5 Training Loop

**Training Script** (`scripts/05_train_ml_estimator.py`):

```python
def train_model(model: nn.Module, train_loader: DataLoader,
                val_loader: DataLoader, config: dict,
                experiment_tracker: ExperimentTracker):
    """
    Standard PyTorch training loop with:
    - MSE/MAE loss
    - Gradient clipping
    - Learning rate scheduling
    - Early stopping
    - Checkpointing
    - Experiment logging
    """
    ...
```

**Loss Functions:**

- Primary: `MSELoss` (mean squared error)
- Secondary: `L1Loss` (mean absolute error) for robustness
- Multi-task: Weighted combination for predicting multiple targets
**Evaluation Metrics:**
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- R² (Coefficient of Determination)

### 8.6 Multi-Step Prediction

**Strategy:** The model predicts the next `prediction_horizon` time steps directly (direct multi-output) rather than iteratively (recursive), to avoid error accumulation.
**Output targets:**

- `predicted_vehicle_count` (for each future step)
- `predicted_occupancy_ratio` (for each future step)
- `predicted_congestion_level` (for each future step)

---

## 9. Inference Pipeline

### 9.1 Orchestrator Design

**Main Orchestrator** (`src/inference/orchestrator.py`):

```python
class InferenceOrchestrator:
    def __init__(self, config: dict):
        # Initialize all components
        self.detector = self._load_detector(config)
        self.segmenter = self._load_segmenter(config)
        self.tracker = self._load_tracker(config)
        self.road_mask_manager = self._load_road_mask(config)
        self.on_road_filter = self._load_filter(config)
        self.feature_extractor = self._load_feature_extractor(config)
        self.ml_estimator = self._load_ml_estimator(config)
        self.mode = config["mode"]  # detect | segment | predict
        self.target_fps = config.get("target_fps", None)
        self.density_window_size = config.get("density_window_size", 30)
    def run(self, video_source: str, output_path: str | None = None):
        """
        Run the full inference pipeline on a video source.
        Pipeline stages (execution depends on mode):
        1. Read frame
        2. Detect/Segment vehicles
        3. Track vehicles
        4. Filter on-road vehicles
        5. Accumulate features over N-frame window
        6. Extract aggregated features when window is full
        7. Predict future density (if predict mode)
        8. Output results
        """
        ...
```

### 9.2 Fixed-Frame-Rate Inference

**Strategy:** Maintain a consistent output FPS regardless of processing speed:

```python
class FixedFPSInference:
    def __init__(self, target_fps: int = 30):
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
    def run(self, pipeline: Callable, video_source: str):
        """
        Run pipeline at fixed FPS:
        - If processing is faster than target FPS, sleep to maintain rate
        - If processing is slower, skip frames to maintain rate
        - Log actual FPS achieved
        """
        ...
```

### 9.3 Inter-Component Communication

**Queue System** (`src/inference/queue.py`):

```python
class PipelineQueue:
    """Thread-safe queue with overflow handling."""
    def __init__(self, maxsize: int = 10, overflow_policy: str = "drop_oldest"):
        ...
    def put(self, item: Any, timeout: float = 1.0):
        ...
    def get(self, timeout: float = 1.0) -> Any:
        ...
```

**Data Types Between Stages:**

```python
@dataclass
class DetectionResult:
    frame_idx: int
    timestamp: float
    boxes: np.ndarray
    confidences: np.ndarray
    class_ids: np.ndarray
    masks: np.ndarray | None  # Only in segmentation mode
@dataclass
class TrackingResult:
    frame_idx: int
    timestamp: float
    tracked_objects: list[dict]
@dataclass
class FilteredResult:
    frame_idx: int
    timestamp: float
    on_road_vehicles: list[dict]
    occupancy_ratio: float
    vehicle_count: int
@dataclass
class WindowFeatures:
    window_start: int
    window_end: int
    features: np.ndarray  # Aggregated features for the window
@dataclass
class PredictionResult:
    window_end: int
    timestamp: float
    current_density: float
    predicted_density: np.ndarray  # N-step ahead
    confidence: np.ndarray
```

---

## 10. Model Optimization

### 10.1 ONNX Export

**Export Script** (`src/optimization/onnx_export.py`):

```python
def export_to_onnx(model_path: str, output_path: str,
                   input_shape: tuple = (1, 3, 640, 640),
                   opset_version: int = 17,
                   dynamic_axes: dict | None = None):
    """
    Export PyTorch/Ultralytics model to ONNX format.
    Supports detection and segmentation models.
    """
    ...
```

**Dynamic Axes:**

```python
dynamic_axes = {
    "input": {0: "batch", 2: "height", 3: "width"},
    "output": {0: "batch", 1: "num_detections"}
}
```

### 10.2 Optimization Benchmarking

**Benchmark** (`src/optimization/benchmark.py`):

```python
def benchmark_model(model_path: str, model_format: str,  # pytorch | onnx
                    input_shape: tuple, num_iterations: int = 1000,
                    warmup_iterations: int = 100) -> dict:
    """
    Benchmark model inference speed.
    Returns: {
        "mean_latency_ms": float,
        "std_latency_ms": float,
        "p50_latency_ms": float,
        "p95_latency_ms": float,
        "p99_latency_ms": float,
        "fps": float,
        "gpu_memory_mb": float
    }
    """
    ...
```

---

## 11. Evaluation & Benchmarking

### 11.1 Metrics

**Metric Implementations** (`src/evaluation/metrics.py`):

```python
class MetricsCalculator:
    @staticmethod
    def fps(processed_frames: int, elapsed_seconds: float) -> float:
        ...
    @staticmethod
    def latency_ms(latencies: list[float]) -> dict[str, float]:
        """Returns mean, std, p50, p95, p99."""
        ...
    @staticmethod
    def mae(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Mean Absolute Error for vehicle counts."""
        ...
    @staticmethod
    def rmse(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Root Mean Square Error for density predictions."""
        ...
    @staticmethod
    def mape(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Mean Absolute Percentage Error."""
        ...
    @staticmethod
    def r_squared(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Coefficient of Determination."""
        ...
    @staticmethod
    def detection_metrics(predictions: list[dict], ground_truth: list[dict],
                          iou_threshold: float = 0.5) -> dict:
        """Precision, Recall, mAP@50, mAP@50:95."""
        ...
```

### 11.2 Benchmark Runner

**Benchmark** (`src/evaluation/benchmark.py`):

```python
class BenchmarkRunner:
    def __init__(self, config: dict):
        ...
    def run_model_benchmark(self, model_configs: list[dict],
                            test_scenes: list[str]) -> pd.DataFrame:
        """
        Benchmark multiple model configurations across test scenes.
        For each (model, scene) pair, measure:
        - FPS
        - Latency (mean, p50, p95, p99)
        - MAE in vehicle counts
        - Detection mAP (if ground truth available)
        - GPU memory usage
        """
        ...
    def run_predictor_benchmark(self, estimator_configs: list[dict],
                                test_data: list[str]) -> pd.DataFrame:
        """
        Benchmark ML estimators.
        For each (estimator, data) pair, measure:
        - RMSE
        - MAE
        - MAPE
        - R²
        - Inference latency
        """
        ...
    def run_environmental_comparison(self, model_config: dict,
                                     scene_types: list[str]) -> pd.DataFrame:
        """
        Compare model performance across environment types.
        Scene types: highway, urban, suburban, intersection
        """
        ...
```

### 11.3 Visualization

**Graph Generation** (`src/evaluation/visualization.py`):

```python
class VisualizationGenerator:
    def __init__(self, output_dir: str = "experiments/results/"):
        ...
    def plot_fps_comparison(self, results: pd.DataFrame, save_path: str):
        """Bar chart: FPS across model variants."""
        ...
    def plot_latency_distribution(self, latencies: dict, save_path: str):
        """Box plot: Latency distribution with percentiles."""
        ...
    def plot_mae_comparison(self, results: pd.DataFrame, save_path: str):
        """Bar chart: MAE in vehicle counts across models."""
        ...
    def plot_prediction_accuracy(self, predictions: dict, save_path: str):
        """Time series: Actual vs predicted density over time."""
        ...
    def plot_rmse_by_model(self, results: pd.DataFrame, save_path: str):
        """Bar chart: RMSE across ML estimator architectures."""
        ...
    def plot_environment_comparison(self, results: pd.DataFrame, save_path: str):
        """Grouped bar chart: Performance by environment type."""
        ...
    def plot_optimization_impact(self, before: dict, after: dict, save_path: str):
        """Grouped bar chart: Before/after optimization metrics."""
        ...
```

---

## 12. Experiment Tracking

### 12.1 Unified Tracker Interface

**Interface** (`src/experiment/tracker.py`):

```python
class ExperimentTracker:
    def __init__(self, backends: list[str] = ["tensorboard", "wandb"],
                 experiment_name: str = "default",
                 config: dict | None = None):
        self.loggers = []
        for backend in backends:
            if backend == "tensorboard":
                self.loggers.append(TensorBoardLogger(experiment_name))
            elif backend == "wandb":
                self.loggers.append(WandBLogger(experiment_name, config))
    def log_params(self, params: dict):
        for logger in self.loggers:
            logger.log_params(params)
    def log_metrics(self, metrics: dict, step: int | None = None):
        for logger in self.loggers:
            logger.log_metrics(metrics, step)
    def log_image(self, image: np.ndarray, caption: str, step: int | None = None):
        for logger in self.loggers:
            logger.log_image(image, caption, step)
    def log_model(self, model_path: str, metadata: dict | None = None):
        for logger in self.loggers:
            logger.log_model(model_path, metadata)
    def finish(self):
        for logger in self.loggers:
            logger.finish()
```

### 12.2 TensorBoard Integration

**Logger** (`src/experiment/tensorboard_logger.py`):

- Logs to `experiments/logs/tensorboard/<experiment_name>/`
- Supports scalars, images, histograms, graphs
- Configurable via `--tensorboard` flag

### 12.3 Weights & Biases Integration

**Logger** (`src/experiment/wandb_logger.py`):

- Logs to W&B cloud (requires `wandb login`)
- Supports all metric types, model artifacts, interactive plots
- Configurable via `--wandb` flag
- Offline mode supported for air-gapped environments

### 12.4 Logged Metrics

**During CV Model Training:**

- Per-epoch: train_loss, val_loss, mAP@50, mAP@50:95, precision, recall
- Per-batch: batch_loss, learning rate
- Images: prediction overlays on validation set every 10 epochs
- Hyperparameters: full config
**During ML Estimator Training:**
- Per-epoch: train_loss, val_loss, train_rmse, val_rmse, val_mape
- Per-batch: batch_loss, learning rate
- Predictions: actual vs predicted plots every 10 epochs
**During Inference Benchmarking:**
- FPS, latency (mean, p50, p95, p99)
- GPU memory usage
- MAE in vehicle counts
- Environmental breakdown

---

## 13. Configuration System

### 13.1 YAML-Based Configs

**Config Loader** (`src/utils/config.py`):

```python
class Config:
    def __init__(self, config_path: str, base_config_path: str | None = None):
        """
        Load config from YAML, optionally merging with base config.
        Validates required fields.
        """
        ...
    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        ...
    def get(self, key: str, default: Any = None) -> Any:
        ...
    def validate(self):
        """Validate required fields and value ranges."""
        ...
```

### 13.2 Configuration Hierarchy

```
base.yaml (defaults)
    │
    ├── models/detection/yolov8s.yaml
    │       │
    │       └── experiments/baseline_detection.yaml
    │
    ├── models/segmentation/yolov11s-seg.yaml
    │       │
    │       └── experiments/baseline_segmentation.yaml
    │
    └── models/ml_estimator/lstm.yaml
            │
            └── experiments/predictive_lstm.yaml
```

### 13.3 CLI Argument Override

All scripts support CLI overrides:

```bash
python scripts/03_train_detection.py \
    --config configs/experiments/baseline_detection.yaml \
    --model.name yolov8s \
    --training.epochs 150 \
    --training.batch_size 32 \
    --device cuda:0
```

---

## 14. Testing Strategy

### 14.1 Unit Tests

| Test File | What It Tests |
|-----------|--------------|
| `test_data/test_raw_loader.py` | Video loading, frame extraction |
| `test_data/test_preprocess.py` | Frame resizing, normalization |
| `test_data/test_dataset.py` | Dataset **getitem**, transforms |
| `test_models/test_detection.py` | Detection model forward pass, output shapes |
| `test_models/test_segmentation.py` | Segmentation model forward pass, mask shapes |
| `test_models/test_ml_estimator.py` | LSTM/GRU/TCN forward pass, output shapes |
| `test_tracking/test_tracker.py` | Tracking update, ID persistence |
| `test_spatial_logic/test_road_mask.py` | Road mask loading, generation |
| `test_spatial_logic/test_intersection.py` | IoU computation correctness |
| `test_spatial_logic/test_filtering.py` | On-road classification logic |
| `test_features/test_extractor.py` | Feature extraction correctness |
| `test_inference/test_orchestrator.py` | Pipeline end-to-end (mocked) |
| `test_evaluation/test_metrics.py` | Metric computation correctness |

### 14.2 Integration Tests

| Test | Description |
|------|------------|
| End-to-end detection | Run detection on sample video, verify output format |
| End-to-end segmentation | Run segmentation on sample video, verify masks |
| End-to-end pipeline | Run full pipeline on short video, verify density output |
| ML estimator training | Train on synthetic data, verify convergence |

### 14.3 Test Fixtures

**Synthetic Data Generation:**

- Generate synthetic frames with known vehicle positions
- Generate synthetic road masks
- Generate synthetic telemetry data with known patterns
- Allows deterministic testing without real data

---

## 15. Dependencies

### 15.1 Core Dependencies

```toml
[project.dependencies]
# Computer Vision
ultralytics = ">=8.2.0"          # YOLO models, tracking
opencv-python = ">=4.9.0"        # Image/video processing
pillow = ">=10.0.0"              # Image handling
# Deep Learning
torch = ">=2.1.0"                # PyTorch
torchvision = ">=0.16.0"         # Torchvision utilities
# Data Processing
numpy = ">=1.26.0"               # Numerical operations
pandas = ">=2.1.0"               # Data manipulation
pyarrow = ">=14.0.0"             # Parquet support
roboflow = ">=1.1.0"             # Roboflow API client
# Experiment Tracking
tensorboard = ">=2.15.0"         # TensorBoard
wandb = ">=0.16.0"               # Weights & Biases
# Configuration
pyyaml = ">=6.0.1"               # YAML parsing
# Visualization
matplotlib = ">=3.8.0"           # Plotting
seaborn = ">=0.13.0"             # Statistical plots
# Utilities
tqdm = ">=4.66.0"                # Progress bars
rich = ">=13.7.0"                # Rich CLI output
loguru = ">=0.7.0"               # Logging
# Model Optimization
onnx = ">=1.15.0"                # ONNX format
onnxruntime = ">=1.16.0"         # ONNX inference
# Testing
pytest = ">=7.4.0"               # Testing framework
pytest-cov = ">=4.1.0"           # Coverage reporting
```

### 15.2 Optional Dependencies

```toml
[project.optional-dependencies]
dev = ["ruff", "mypy", "pre-commit"]
```

### 15.3 Python Version

```
requires-python = ">=3.10"
```

Python 3.10+ required for:

- `match` statements (cleaner config parsing)
- `|` union type syntax
- Better type hints

---

## 16. Future Work

### 16.1 Vision Transformer Segmentation

**Purpose:** Explore ViT-based segmentation for improved accuracy on complex road scenes.
**Architecture:**

- Backbone: ViT-S or ViT-B (pre-trained on ImageNet-21K)
- Decoder: Mask2Former or SETR-style decoder
- Input: 512x512 or 640x640
**Configuration:**

```yaml
# configs/models/segmentation/vit-seg.yaml (future)
model:
  name: "vit-seg"
  backbone: "vit_b"
  decoder: "mask2former"
  pretrained: true
training:
  epochs: 150
  batch_size: 8
  image_size: 512
  optimizer: "AdamW"
  lr: 0.0001
  lr_scheduler: "cosine"
  warmup_epochs: 10
  weight_decay: 0.01
  patience: 30
```

### 16.2 Parallel Pipeline Execution

**Purpose:** Decompose the inference pipeline into stages that run asynchronously to maximize hardware utilization.
**Planned Architecture:**

```
Stage 1: Video I/O (reads frames, puts in queue)
    │
    ▼
Stage 2: Detection/Segmentation (consumes frames, produces detections)
    │
    ▼
Stage 3: Tracking (consumes detections, produces tracked objects)
    │
    ▼
Stage 4: Spatial Logic (consumes tracked objects, produces filtered data)
    │
    ▼
Stage 5: Feature Extraction + ML Estimator (consumes filtered data, produces predictions)
    │
    ▼
Stage 6: Output (writes results, visualization)
```

**Parallelization Strategy:**

| Stage | Parallelizable? | Method |
|-------|----------------|--------|
| Video I/O | Yes | Separate thread |
| Detection | No (GPU-bound) | Main thread |
| Tracking | No (depends on detection) | Main thread |
| Spatial Logic | Yes | Separate thread |
| Feature Extraction | Yes | Separate thread |
| ML Estimator | Yes | Separate thread |
| Output | Yes | Separate thread |

### 16.3 Dynamic Road Masking

- Handle construction zones and temporary lane closures
- Update road mask in real-time based on detected changes
- Integrate temporal consistency checks

### 16.4 Environmental Condition Integration

- Weather detection (rain, fog, snow) as input features
- Lighting condition adaptation (day, dusk, night)
- Camera quality assessment and auto-adjustment

### 16.5 Multi-Camera Fusion

- Handle occlusions across multiple camera views
- Cross-camera vehicle re-identification
- Unified density estimation across camera network
