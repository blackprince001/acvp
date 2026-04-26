"""
Pixel occupancy computation.

Computes the fraction of road pixels covered by vehicle masks.
"""

from __future__ import annotations

import numpy as np


def compute_occupancy(
  vehicle_masks: list[np.ndarray],
  road_mask: np.ndarray,
) -> float:
  """Compute road pixel occupancy ratio.

  Args:
      vehicle_masks: List of boolean/uint8 arrays (H, W), one per vehicle.
      road_mask: Boolean/uint8 array (H, W) representing drivable area.

  Returns:
      Ratio of road pixels covered by at least one vehicle, in [0, 1].
      Returns 0.0 when road_mask is empty or vehicle_masks is empty.
  """
  road = road_mask.astype(bool)
  road_pixels = road.sum()
  if road_pixels == 0 or not vehicle_masks:
    return 0.0

  union = np.zeros(road.shape, dtype=bool)
  for m in vehicle_masks:
    union |= m.astype(bool)

  covered = np.logical_and(union, road).sum()
  return float(covered) / float(road_pixels)
