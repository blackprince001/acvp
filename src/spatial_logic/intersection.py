"""
IoU-based overlap computation between detections and road mask.

Two public functions:
- compute_mask_road_iou  – vehicle binary mask vs road mask
- compute_box_road_iou   – bounding box (xyxy) vs road mask
"""

from __future__ import annotations

import numpy as np


def compute_mask_road_iou(vehicle_mask: np.ndarray, road_mask: np.ndarray) -> float:
  """Compute IoU between a vehicle segmentation mask and the road mask.

  Args:
      vehicle_mask: Boolean/uint8 array of shape (H, W).
      road_mask: Boolean/uint8 array of shape (H, W), same size.

  Returns:
      Overlap ratio in [0, 1].  Returns 0.0 when both masks are empty.
  """
  v = vehicle_mask.astype(bool)
  r = road_mask.astype(bool)
  intersection = np.logical_and(v, r).sum()
  union = np.logical_or(v, r).sum()
  if union == 0:
    return 0.0
  return float(intersection) / float(union)


def compute_box_road_iou(box: np.ndarray | list[float], road_mask: np.ndarray) -> float:
  """Compute the fraction of a bounding box that overlaps the road mask.

  The metric is:  road pixels inside the box / total pixels inside the box.
  This is intentionally *not* a symmetric IoU — it answers "how much of
  this vehicle is on the road?" which is more useful for filtering.

  Args:
      box: Bounding box [x1, y1, x2, y2] in pixel coordinates (xyxy).
      road_mask: Boolean/uint8 array of shape (H, W).

  Returns:
      Overlap ratio in [0, 1].  Returns 0.0 for zero-area boxes.
  """
  x1, y1, x2, y2 = (int(round(v)) for v in box)
  h, w = road_mask.shape[:2]

  # Clamp to image bounds
  x1 = max(0, min(x1, w))
  x2 = max(0, min(x2, w))
  y1 = max(0, min(y1, h))
  y2 = max(0, min(y2, h))

  if x2 <= x1 or y2 <= y1:
    return 0.0

  roi = road_mask[y1:y2, x1:x2].astype(bool)
  total = roi.size
  if total == 0:
    return 0.0
  return float(roi.sum()) / float(total)
