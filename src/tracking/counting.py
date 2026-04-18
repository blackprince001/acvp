"""
Redundant counting prevention module.

Provides zone-based counting with unique ID tracking to prevent
double-counting vehicles as they traverse the scene.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .base import Track


class CrossingDirection(Enum):
  """Direction of line crossing."""

  NONE = "none"
  POSITIVE = "positive"  # Crossed from negative to positive side
  NEGATIVE = "negative"  # Crossed from positive to negative side


@dataclass
class CountingLine:
  """Defines a counting line for vehicle counting.

  A counting line is defined by two points and has a direction.
  Vehicles crossing from left to right (relative to the line direction)
  are counted as positive crossings.

  Attributes:
      line_id: Unique identifier for this counting line.
      start: Start point (x, y) of the line.
      end: End point (x, y) of the line.
      name: Optional human-readable name for the line.
  """

  line_id: int
  start: np.ndarray
  end: np.ndarray
  name: str = ""

  def __post_init__(self) -> None:
    self.start = np.asarray(self.start, dtype=float)
    self.end = np.asarray(self.end, dtype=float)

  def get_side(self, point: np.ndarray) -> int:
    """Determine which side of the line a point is on.

    Args:
        point: Point (x, y) to check.

    Returns:
        Positive value if point is on the right side of the line
        (when looking from start to end), negative if on the left,
        and 0 if on the line.
    """
    # Cross product to determine side
    d = (point[0] - self.start[0]) * (self.end[1] - self.start[1]) - (
      point[1] - self.start[1]
    ) * (self.end[0] - self.start[0])
    return int(np.sign(d))


@dataclass
class CountingZone:
  """Defines a counting zone (polygon region) for vehicle counting.

  Vehicles entering the zone for the first time are counted.

  Attributes:
      zone_id: Unique identifier for this zone.
      vertices: Polygon vertices as array of shape (N, 2).
      name: Optional human-readable name for the zone.
  """

  zone_id: int
  vertices: np.ndarray
  name: str = ""

  def __post_init__(self) -> None:
    self.vertices = np.asarray(self.vertices, dtype=float)

  def contains(self, point: np.ndarray) -> bool:
    """Check if a point is inside the zone using ray casting.

    Args:
        point: Point (x, y) to check.

    Returns:
        True if the point is inside the polygon.
    """
    x, y = point
    n = len(self.vertices)
    inside = False

    j = n - 1
    for i in range(n):
      xi, yi = self.vertices[i]
      xj, yj = self.vertices[j]

      if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
        inside = not inside
      j = i

    return inside


@dataclass
class CountingResult:
  """Result of a counting operation.

  Attributes:
      line_crossings: Dict mapping line_id to count of crossings.
      zone_entries: Dict mapping zone_id to count of entries.
      total_unique_ids: Total number of unique track IDs seen.
      positive_crossings: Count of positive direction crossings.
      negative_crossings: Count of negative direction crossings.
  """

  line_crossings: dict[int, int] = field(default_factory=dict)
  zone_entries: dict[int, int] = field(default_factory=dict)
  total_unique_ids: int = 0
  positive_crossings: int = 0
  negative_crossings: int = 0


class VehicleCounter:
  """Zone-based vehicle counter with redundant counting prevention.

  Tracks unique vehicle IDs and prevents double-counting by maintaining
  a history of which tracks have crossed each counting line or entered
  each counting zone.

  Args:
      config: Optional configuration dict with counter settings.

  Example::

      counter = VehicleCounter()
      counter.add_line(CountingLine(1, [0, 100], [640, 100], "entrance"))
      counter.add_zone(CountingZone(1, [[0, 0], [100, 0], [100, 100], [0, 100]], "roi"))

      for frame_tracks in video_tracks:
          result = counter.update(frame_tracks)
          print(f"Total crossings: {result.line_crossings}")
  """

  def __init__(self, config: dict[str, Any] | None = None) -> None:
    self.config = config or {}

    # Counting elements
    self._lines: dict[int, CountingLine] = {}
    self._zones: dict[int, CountingZone] = {}

    # Tracking state for redundancy prevention
    self._track_line_sides: dict[
      int, dict[int, int]
    ] = {}  # track_id -> line_id -> side
    self._track_line_crossed: dict[
      int, set[int]
    ] = {}  # track_id -> set of crossed lines
    self._track_zone_entered: dict[
      int, set[int]
    ] = {}  # track_id -> set of entered zones
    self._all_seen_ids: set[int] = set()

    # Cumulative counts
    self._line_counts: dict[int, int] = {}
    self._zone_counts: dict[int, int] = {}
    self._positive_crossings: int = 0
    self._negative_crossings: int = 0

  def add_line(self, line: CountingLine) -> None:
    """Add a counting line.

    Args:
        line: The CountingLine to add.
    """
    self._lines[line.line_id] = line
    self._line_counts[line.line_id] = 0

  def add_zone(self, zone: CountingZone) -> None:
    """Add a counting zone.

    Args:
        zone: The CountingZone to add.
    """
    self._zones[zone.zone_id] = zone
    self._zone_counts[zone.zone_id] = 0

  def update(self, tracks: list[Track]) -> CountingResult:
    """Process tracks and update counts.

    Checks each track against all counting lines and zones,
    updating counts only for first-time crossings/entries.

    Args:
        tracks: List of Track objects from the current frame.

    Returns:
        CountingResult with current cumulative counts.
    """
    for track in tracks:
      track_id = track.track_id
      centroid = track.centroid

      # Track all unique IDs
      self._all_seen_ids.add(track_id)

      # Initialize tracking state for new tracks
      if track_id not in self._track_line_sides:
        self._track_line_sides[track_id] = {}
        self._track_line_crossed[track_id] = set()
        self._track_zone_entered[track_id] = set()

      # Check line crossings
      for line_id, line in self._lines.items():
        current_side = line.get_side(centroid)

        if current_side == 0:
          # On the line, skip
          continue

        if line_id in self._track_line_sides[track_id]:
          prev_side = self._track_line_sides[track_id][line_id]

          # Check if crossed (side changed)
          if prev_side != 0 and prev_side != current_side:
            # Only count if not already crossed
            if line_id not in self._track_line_crossed[track_id]:
              self._track_line_crossed[track_id].add(line_id)
              self._line_counts[line_id] += 1

              # Track crossing direction
              if current_side > 0:
                self._positive_crossings += 1
              else:
                self._negative_crossings += 1

        # Update side for next frame
        self._track_line_sides[track_id][line_id] = current_side

      # Check zone entries
      for zone_id, zone in self._zones.items():
        if zone.contains(centroid):
          # Only count first entry
          if zone_id not in self._track_zone_entered[track_id]:
            self._track_zone_entered[track_id].add(zone_id)
            self._zone_counts[zone_id] += 1

    return CountingResult(
      line_crossings=self._line_counts.copy(),
      zone_entries=self._zone_counts.copy(),
      total_unique_ids=len(self._all_seen_ids),
      positive_crossings=self._positive_crossings,
      negative_crossings=self._negative_crossings,
    )

  def get_counts(self) -> CountingResult:
    """Get current cumulative counts without processing new tracks.

    Returns:
        Current CountingResult.
    """
    return CountingResult(
      line_crossings=self._line_counts.copy(),
      zone_entries=self._zone_counts.copy(),
      total_unique_ids=len(self._all_seen_ids),
      positive_crossings=self._positive_crossings,
      negative_crossings=self._negative_crossings,
    )

  def reset(self) -> None:
    """Reset all counts and tracking state."""
    self._track_line_sides.clear()
    self._track_line_crossed.clear()
    self._track_zone_entered.clear()
    self._all_seen_ids.clear()

    for line_id in self._line_counts:
      self._line_counts[line_id] = 0
    for zone_id in self._zone_counts:
      self._zone_counts[zone_id] = 0

    self._positive_crossings = 0
    self._negative_crossings = 0

  def remove_stale_tracks(self, active_track_ids: set[int]) -> None:
    """Remove tracking state for tracks no longer active.

    Call this periodically to prevent memory growth for long videos.

    Args:
        active_track_ids: Set of currently active track IDs.
    """
    stale_ids = set(self._track_line_sides.keys()) - active_track_ids

    for track_id in stale_ids:
      self._track_line_sides.pop(track_id, None)
      self._track_line_crossed.pop(track_id, None)
      self._track_zone_entered.pop(track_id, None)

  def __repr__(self) -> str:
    return (
      f"VehicleCounter(lines={len(self._lines)}, zones={len(self._zones)}, "
      f"unique_ids={len(self._all_seen_ids)})"
    )
