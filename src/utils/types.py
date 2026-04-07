from dataclasses import dataclass
import numpy as np

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
