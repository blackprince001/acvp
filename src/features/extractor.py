"""
Feature extractor.

Extracts 10 per-frame features from a FilteredResult + road mask.

Feature vector layout (index → name):
  0  vehicle_count
  1  occupancy_ratio
  2  mean_speed
  3  mean_direction
  4  density
  5  flow
  6  congestion_index
  7  stopped_vehicle_ratio
  8  speed_variance
  9  direction_variance
"""

from __future__ import annotations

import numpy as np

from src.utils.types import FilteredResult

FEATURE_NAMES = [
  "vehicle_count",
  "occupancy_ratio",
  "mean_speed",
  "mean_direction",
  "density",
  "flow",
  "congestion_index",
  "stopped_vehicle_ratio",
  "speed_variance",
  "direction_variance",
]
N_FEATURES = len(FEATURE_NAMES)


class FeatureExtractor:
  """Extract a 10-element feature vector from per-frame pipeline output.

  Args:
      max_speed: Reference maximum speed (px/frame) for congestion index.
      stop_threshold: Speed below which a vehicle is considered stopped.
  """

  def __init__(self, max_speed: float = 50.0, stop_threshold: float = 1.0) -> None:
    self.max_speed = max_speed
    self.stop_threshold = stop_threshold

  def extract(
    self,
    filtered: FilteredResult,
    road_mask: np.ndarray,
    flow: float = 0.0,
  ) -> np.ndarray:
    """Compute the 10-feature vector for one frame.

    Args:
        filtered: Output of OnRoadFilter.filter().
        road_mask: Boolean (H, W) road mask used for density.
        flow: Vehicle flow count for this frame (from VehicleCounter).

    Returns:
        Float64 ndarray of shape (10,).
    """
    vehicles = filtered.on_road_vehicles
    n = len(vehicles)

    speeds = (
      np.array([v["speed"] for v in vehicles], dtype=float) if n else np.array([])
    )
    directions = (
      np.array([v["direction"] for v in vehicles], dtype=float) if n else np.array([])
    )

    road_area = float(road_mask.astype(bool).sum())

    mean_speed = float(speeds.mean()) if n else 0.0
    mean_direction = float(directions.mean()) if n else 0.0
    density = n / road_area if road_area > 0 else 0.0

    # congestion: occupancy scaled by how slow traffic is (0 when no vehicles)
    if n > 0 and self.max_speed > 0:
      slowness = 1.0 - min(mean_speed / self.max_speed, 1.0)
      congestion_index = float(np.clip(filtered.occupancy_ratio * slowness, 0.0, 1.0))
    else:
      congestion_index = 0.0

    stopped_ratio = float((speeds < self.stop_threshold).sum() / n) if n else 0.0
    speed_var = float(speeds.var()) if n else 0.0
    direction_var = float(directions.var()) if n else 0.0

    return np.array(
      [
        float(n),
        filtered.occupancy_ratio,
        mean_speed,
        mean_direction,
        density,
        float(flow),
        congestion_index,
        stopped_ratio,
        speed_var,
        direction_var,
      ],
      dtype=np.float64,
    )
