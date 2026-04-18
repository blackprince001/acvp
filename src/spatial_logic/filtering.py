"""
On-road classification filter (T-052).

Classifies tracked vehicles as on-road or off-road using overlap threshold
and an optional minimum-speed gate.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.utils.types import FilteredResult

from .intersection import compute_box_road_iou, compute_mask_road_iou


class OnRoadFilter:
  """Classify tracked vehicles as on-road or off-road.

  Args:
      overlap_threshold: Minimum overlap ratio to be considered on-road.
          Defaults to 0.5.
      min_speed: If > 0, stationary vehicles (speed < min_speed px/frame)
          are excluded even when their overlap is sufficient.  Defaults to 0
          (disabled).
  """

  def __init__(
    self,
    overlap_threshold: float = 0.5,
    min_speed: float = 0.0,
  ) -> None:
    self.overlap_threshold = overlap_threshold
    self.min_speed = min_speed

  # ------------------------------------------------------------------

  def filter(
    self,
    tracking_output: Any,  # TrackingOutput from src.tracking.base
    road_mask: np.ndarray,
    frame_idx: int = 0,
    timestamp: float = 0.0,
  ) -> FilteredResult:
    """Filter tracked objects to those on the road.

    Args:
        tracking_output: A TrackingOutput (or any object with a .tracks
            list of Track objects).
        road_mask: Boolean array (H, W) representing drivable area.
        frame_idx: Frame index for the result.
        timestamp: Timestamp for the result.

    Returns:
        FilteredResult with on_road_vehicles, occupancy_ratio, vehicle_count.
    """
    on_road: list[dict] = []

    for track in tracking_output.tracks:
      # Speed gate (optional)
      if self.min_speed > 0 and track.speed < self.min_speed:
        continue

      # Use segmentation mask if available, else bounding box
      if hasattr(track, "mask") and track.mask is not None:
        overlap = compute_mask_road_iou(track.mask, road_mask)
      else:
        overlap = compute_box_road_iou(track.bbox, road_mask)

      if overlap >= self.overlap_threshold:
        on_road.append(
          {
            "track_id": track.track_id,
            "bbox": track.bbox,
            "centroid": track.centroid,
            "confidence": track.confidence,
            "class_id": track.class_id,
            "speed": track.speed,
            "direction": track.direction,
            "overlap": overlap,
          }
        )

    # Simple occupancy: fraction of road pixels covered by on-road vehicles
    occupancy = _compute_occupancy_from_boxes([v["bbox"] for v in on_road], road_mask)

    return FilteredResult(
      frame_idx=frame_idx,
      timestamp=timestamp,
      on_road_vehicles=on_road,
      occupancy_ratio=occupancy,
      vehicle_count=len(on_road),
    )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _compute_occupancy_from_boxes(
  boxes: list[np.ndarray], road_mask: np.ndarray
) -> float:
  """Union of box pixels on road / total road pixels."""
  road_pixels = road_mask.astype(bool).sum()
  if road_pixels == 0:
    return 0.0

  h, w = road_mask.shape[:2]
  union = np.zeros((h, w), dtype=bool)
  for box in boxes:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 > x1 and y2 > y1:
      union[y1:y2, x1:x2] = True

  covered = np.logical_and(union, road_mask).sum()
  return float(covered) / float(road_pixels)
