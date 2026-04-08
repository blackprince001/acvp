"""
Base tracker interface.

All tracker implementations (BoT-SORT, ByteTrack, etc.) must subclass this
and implement the core methods: update(), get_tracks(), and reset().
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Track:
  """Represents a single tracked object.

  Attributes:
      track_id: Unique identifier for this track across frames.
      bbox: Bounding box in xyxy format [x1, y1, x2, y2].
      confidence: Detection confidence score.
      class_id: Class ID of the detected object.
      centroid: Center point (x, y) of the bounding box.
      velocity: Movement velocity (vx, vy) in pixels per frame.
      direction: Movement direction in degrees (0-360, 0=right, 90=down).
      speed: Movement speed (magnitude of velocity) in pixels per frame.
      age: Number of frames this track has been active.
      hits: Number of consecutive frames with detections.
      time_since_update: Frames since last detection match.
      history: List of past centroids for trajectory analysis.
  """

  track_id: int
  bbox: np.ndarray
  confidence: float
  class_id: int
  centroid: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
  velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
  direction: float = 0.0
  speed: float = 0.0
  age: int = 1
  hits: int = 1
  time_since_update: int = 0
  history: list[np.ndarray] = field(default_factory=list)

  def __post_init__(self) -> None:
    """Compute centroid from bbox if not provided."""
    if self.centroid is None or (
      isinstance(self.centroid, np.ndarray) and self.centroid.sum() == 0
    ):
      x1, y1, x2, y2 = self.bbox
      self.centroid = np.array([(x1 + x2) / 2, (y1 + y2) / 2])


@dataclass
class TrackingOutput:
  """Output from a single tracking update.

  Attributes:
      frame_idx: Frame index in the video sequence.
      tracks: List of active tracks in this frame.
      lost_tracks: List of track IDs that were lost in this frame.
      new_tracks: List of track IDs that were created in this frame.
  """

  frame_idx: int
  tracks: list[Track]
  lost_tracks: list[int] = field(default_factory=list)
  new_tracks: list[int] = field(default_factory=list)


class BaseTracker(ABC):
  """Abstract base class defining the tracker interface.

  Args:
      config: Tracker configuration dict (typically loaded from a YAML config).
  """

  def __init__(self, config: dict[str, Any]) -> None:
    self.config = config
    self._frame_idx = 0
    self._trajectories: dict[int, list[np.ndarray]] = {}

  @abstractmethod
  def update(
    self,
    detections: np.ndarray,
    confidences: np.ndarray,
    class_ids: np.ndarray,
    frame: np.ndarray | None = None,
  ) -> TrackingOutput:
    """Process detections for a single frame and update tracks.

    Args:
        detections: Detection bounding boxes in xyxy format, shape (N, 4).
        confidences: Detection confidence scores, shape (N,).
        class_ids: Detection class IDs, shape (N,).
        frame: Optional raw frame image (some trackers use appearance features).

    Returns:
        TrackingOutput containing the current active tracks and metadata.
    """
    ...

  @abstractmethod
  def reset(self) -> None:
    """Reset the tracker state.

    Clears all active tracks and resets internal state for processing
    a new video sequence.
    """
    ...

  def get_trajectories(self) -> dict[int, list[np.ndarray]]:
    """Get historical trajectories for all tracks.

    Returns:
        Dictionary mapping track IDs to lists of centroid positions.
        Each centroid is an ndarray of shape (2,) representing (x, y).
    """
    return self._trajectories.copy()

  def get_trajectory(self, track_id: int) -> list[np.ndarray] | None:
    """Get the trajectory for a specific track.

    Args:
        track_id: The track ID to look up.

    Returns:
        List of centroid positions for the track, or None if not found.
    """
    return self._trajectories.get(track_id)

  @property
  def frame_idx(self) -> int:
    """Current frame index."""
    return self._frame_idx

  def _update_trajectory(self, track_id: int, centroid: np.ndarray) -> None:
    """Update the trajectory history for a track.

    Args:
        track_id: The track ID to update.
        centroid: The new centroid position.
    """
    if track_id not in self._trajectories:
      self._trajectories[track_id] = []
    self._trajectories[track_id].append(centroid.copy())

  def __repr__(self) -> str:
    name = self.config.get("tracker_type", self.__class__.__name__)
    return f"{self.__class__.__name__}(tracker={name!r}, frame={self._frame_idx})"
