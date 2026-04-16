"""
Spatial logic layer.

Provides road mask management, IoU-based overlap computation,
on-road vehicle filtering, and pixel occupancy estimation.
"""

from .filtering import OnRoadFilter
from .intersection import compute_box_road_iou, compute_mask_road_iou
from .occupancy import compute_occupancy
from .road_mask import RoadMaskManager

__all__ = [
    "OnRoadFilter",
    "RoadMaskManager",
    "compute_box_road_iou",
    "compute_mask_road_iou",
    "compute_occupancy",
]
