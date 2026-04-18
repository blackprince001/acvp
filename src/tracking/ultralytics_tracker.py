"""
Ultralytics tracking wrapper.

Wraps Ultralytics' built-in tracking functionality (BoT-SORT and ByteTrack)
providing a consistent interface for the traffic density estimation pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseTracker, Track, TrackingOutput


class UltralyticsTracker(BaseTracker):
  """Tracker wrapper using Ultralytics built-in tracking.

  Supports both BoT-SORT and ByteTrack tracking algorithms through
  the Ultralytics YOLO model's `.track()` method.

  Args:
      config: Tracker configuration dict with keys:
          - tracker_type: "botsort" or "bytetrack"
          - track_high_thresh: High confidence threshold for tracks
          - track_low_thresh: Low confidence threshold for tracks
          - new_track_thresh: Threshold for creating new tracks
          - track_buffer: Number of frames to keep lost tracks
          - match_thresh: IoU threshold for track matching
          - fuse_score: Whether to fuse confidence scores
      model_path: Path to YOLO model weights for detection + tracking.
      device: Compute device ("cuda", "cpu", "mps").

  Example::

      tracker = UltralyticsTracker(
          config={"tracker_type": "botsort", "track_buffer": 30},
          model_path="yolov8n.pt",
          device="cuda"
      )
      output = tracker.update(detections, confidences, class_ids, frame)
  """

  def __init__(
    self,
    config: dict[str, Any],
    model_path: str | Path = "yolov8n.pt",
    device: str = "cuda",
  ) -> None:
    super().__init__(config)
    self.device = device
    self.model_path = Path(model_path)

    # Tracker parameters from config
    self.tracker_type = config.get("tracker_type", "botsort")
    self.track_high_thresh = config.get("track_high_thresh", 0.5)
    self.track_low_thresh = config.get("track_low_thresh", 0.1)
    self.new_track_thresh = config.get("new_track_thresh", 0.6)
    self.track_buffer = config.get("track_buffer", 30)
    self.match_thresh = config.get("match_thresh", 0.8)
    self.fuse_score = config.get("fuse_score", True)

    # Lazy load model
    self._model = None

    # Track state for movement vectors
    self._prev_centroids: dict[int, np.ndarray] = {}
    self._track_ages: dict[int, int] = {}
    self._track_hits: dict[int, int] = {}

  @property
  def model(self):
    """Lazy load the YOLO model."""
    if self._model is None:
      from ultralytics import YOLO

      self._model = YOLO(str(self.model_path))
    return self._model

  def update(
    self,
    detections: np.ndarray,
    confidences: np.ndarray,
    class_ids: np.ndarray,
    frame: np.ndarray | None = None,
  ) -> TrackingOutput:
    """Process detections for a single frame and update tracks.

    This method runs the Ultralytics tracker on the provided frame.
    If frame is provided, tracking uses detection + tracking mode.
    If only detections are provided, external detections are used.

    Args:
        detections: Detection bounding boxes in xyxy format, shape (N, 4).
        confidences: Detection confidence scores, shape (N,).
        class_ids: Detection class IDs, shape (N,).
        frame: Raw frame image (required for Ultralytics tracking).

    Returns:
        TrackingOutput containing the current active tracks and metadata.

    Raises:
        ValueError: If frame is None (Ultralytics requires the frame).
    """
    if frame is None:
      raise ValueError("UltralyticsTracker requires frame input for tracking")

    self._frame_idx += 1

    # Run tracking using Ultralytics
    results = self.model.track(
      source=frame,
      persist=True,
      tracker=f"{self.tracker_type}.yaml",
      conf=self.track_low_thresh,
      iou=self.match_thresh,
      device=self.device,
      verbose=False,
    )

    # Parse tracking results
    tracks = []
    current_track_ids = set()
    new_track_ids = []

    if results and len(results) > 0:
      result = results[0]
      if result.boxes is not None and result.boxes.id is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)

        for i, track_id in enumerate(track_ids):
          bbox = boxes[i]
          conf = confs[i]
          cls_id = cls_ids[i]

          # Compute centroid
          centroid = np.array(
            [
              (bbox[0] + bbox[2]) / 2,
              (bbox[1] + bbox[3]) / 2,
            ]
          )

          # Compute velocity from previous position
          velocity, direction, speed = self._compute_movement_vector(track_id, centroid)

          # Update track age and hits
          if track_id not in self._track_ages:
            self._track_ages[track_id] = 0
            self._track_hits[track_id] = 0
            new_track_ids.append(track_id)

          self._track_ages[track_id] += 1
          self._track_hits[track_id] += 1

          # Get trajectory history
          history = self._trajectories.get(track_id, []).copy()

          track = Track(
            track_id=int(track_id),
            bbox=bbox,
            confidence=float(conf),
            class_id=int(cls_id),
            centroid=centroid,
            velocity=velocity,
            direction=direction,
            speed=speed,
            age=self._track_ages[track_id],
            hits=self._track_hits[track_id],
            time_since_update=0,
            history=history,
          )
          tracks.append(track)
          current_track_ids.add(track_id)

          # Update trajectory and previous centroid
          self._update_trajectory(track_id, centroid)
          self._prev_centroids[track_id] = centroid

    # Find lost tracks (tracks that were active but not in current frame)
    lost_track_ids = [
      tid for tid in self._prev_centroids.keys() if tid not in current_track_ids
    ]

    return TrackingOutput(
      frame_idx=self._frame_idx,
      tracks=tracks,
      lost_tracks=lost_track_ids,
      new_tracks=new_track_ids,
    )

  def update_with_detections(
    self,
    detections: np.ndarray,
    confidences: np.ndarray,
    class_ids: np.ndarray,
    frame: np.ndarray,
  ) -> TrackingOutput:
    """Process frame with external detections.

    This is an alternative to update() that allows using detections
    from a different detector than the one loaded in the tracker.

    Note: Ultralytics tracking is tightly coupled with detection,
    so this method runs full detection + tracking and filters results.

    Args:
        detections: External detection bounding boxes (not used directly).
        confidences: External detection confidence scores (not used directly).
        class_ids: External detection class IDs (not used directly).
        frame: Raw frame image for tracking.

    Returns:
        TrackingOutput containing the current active tracks.
    """
    # For Ultralytics, we run integrated detection + tracking
    return self.update(detections, confidences, class_ids, frame)

  def _compute_movement_vector(
    self, track_id: int, current_centroid: np.ndarray
  ) -> tuple[np.ndarray, float, float]:
    """Compute movement vector for a track.

    Args:
        track_id: The track ID.
        current_centroid: Current centroid position (x, y).

    Returns:
        Tuple of (velocity, direction, speed):
          - velocity: Movement vector (vx, vy) in pixels per frame.
          - direction: Movement direction in degrees (0-360).
          - speed: Movement speed in pixels per frame.
    """
    if track_id not in self._prev_centroids:
      return np.array([0.0, 0.0]), 0.0, 0.0

    prev_centroid = self._prev_centroids[track_id]
    velocity = current_centroid - prev_centroid

    speed = float(np.linalg.norm(velocity))

    # Compute direction (0=right, 90=down, 180=left, 270=up)
    if speed > 0.1:  # Minimum threshold to avoid noise
      direction = float(np.degrees(np.arctan2(velocity[1], velocity[0])))
      # Convert to 0-360 range
      if direction < 0:
        direction += 360
    else:
      direction = 0.0

    return velocity, direction, speed

  def reset(self) -> None:
    """Reset the tracker state.

    Clears all tracks and resets the YOLO model's internal tracker state.
    """
    self._frame_idx = 0
    self._trajectories.clear()
    self._prev_centroids.clear()
    self._track_ages.clear()
    self._track_hits.clear()

    # Reset Ultralytics tracker state by reloading model
    if self._model is not None:
      self._model.predictor = None

  def __repr__(self) -> str:
    return (
      f"UltralyticsTracker(tracker={self.tracker_type!r}, "
      f"model={self.model_path.name!r}, device={self.device!r})"
    )
