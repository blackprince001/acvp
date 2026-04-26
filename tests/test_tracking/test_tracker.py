"""
Unit tests for tracking module.

Tests cover:
- Base tracker interface and Track dataclass.
- UltralyticsTracker wrapper (with mocked YOLO).
- Movement vector computation (velocity, direction, speed).
- Trajectory retrieval and history.
- VehicleCounter for redundant counting prevention.

These tests are designed to run **without a GPU** and **without downloading
actual YOLO weights** by mocking the Ultralytics ``YOLO`` class.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Make the project root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))


def _make_track_result(
  n_tracks: int = 3, track_ids: list[int] | None = None
) -> MagicMock:
  """Create a fake Ultralytics tracking result with *n_tracks* tracked objects."""
  result = MagicMock()
  result.boxes = MagicMock()

  if track_ids is None:
    track_ids = list(range(1, n_tracks + 1))

  # Create mock tensor-like objects
  boxes_data = np.array(
    [[i * 10, i * 10, i * 10 + 50, i * 10 + 50] for i in range(n_tracks)],
    dtype=np.float32,
  )
  track_ids_data = np.array(track_ids, dtype=np.float32)
  confs_data = np.ones(n_tracks, dtype=np.float32) * 0.9
  cls_data = np.zeros(n_tracks, dtype=np.float32)

  result.boxes.xyxy.cpu().numpy.return_value = boxes_data
  result.boxes.id = MagicMock()
  result.boxes.id.cpu().numpy.return_value = track_ids_data
  result.boxes.conf.cpu().numpy.return_value = confs_data
  result.boxes.cls.cpu().numpy.return_value = cls_data

  return result


def _make_empty_track_result() -> MagicMock:
  """Create a fake result with no tracks."""
  result = MagicMock()
  result.boxes = MagicMock()
  result.boxes.id = None
  return result


def _tracker_config() -> dict:
  return {
    "tracker_type": "botsort",
    "track_high_thresh": 0.5,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.6,
    "track_buffer": 30,
    "match_thresh": 0.8,
    "fuse_score": True,
  }


class TestTrackDataclass:
  """Test the Track dataclass."""

  def test_track_creation(self):
    from src.tracking.base import Track

    bbox = np.array([10, 20, 110, 120])
    track = Track(
      track_id=1,
      bbox=bbox,
      confidence=0.95,
      class_id=0,
    )

    assert track.track_id == 1
    assert np.array_equal(track.bbox, bbox)
    assert track.confidence == 0.95
    assert track.class_id == 0

  def test_track_centroid_computed(self):
    from src.tracking.base import Track

    bbox = np.array([0, 0, 100, 100])
    track = Track(track_id=1, bbox=bbox, confidence=0.9, class_id=0)

    # Centroid should be at (50, 50)
    assert np.allclose(track.centroid, [50, 50])

  def test_track_default_values(self):
    from src.tracking.base import Track

    track = Track(
      track_id=1,
      bbox=np.array([0, 0, 10, 10]),
      confidence=0.9,
      class_id=0,
    )

    assert track.age == 1
    assert track.hits == 1
    assert track.time_since_update == 0
    assert isinstance(track.history, list)


class TestTrackingOutputDataclass:
  """Test the TrackingOutput dataclass."""

  def test_tracking_output_creation(self):
    from src.tracking.base import Track, TrackingOutput

    track = Track(
      track_id=1,
      bbox=np.array([0, 0, 50, 50]),
      confidence=0.9,
      class_id=0,
    )

    output = TrackingOutput(
      frame_idx=10,
      tracks=[track],
      lost_tracks=[5, 6],
      new_tracks=[1],
    )

    assert output.frame_idx == 10
    assert len(output.tracks) == 1
    assert output.lost_tracks == [5, 6]
    assert output.new_tracks == [1]


class TestBaseTrackerInterface:
  """Verify that the abstract interface cannot be instantiated directly."""

  def test_cannot_instantiate_base(self):
    from src.tracking.base import BaseTracker

    with pytest.raises(TypeError):
      BaseTracker({})  # type: ignore[abstract]

  def test_trajectory_methods(self):
    """Test trajectory tracking in a concrete implementation."""
    from src.tracking.base import BaseTracker, TrackingOutput

    # Create a minimal concrete implementation for testing
    class MinimalTracker(BaseTracker):
      def update(self, detections, confidences, class_ids, frame=None):
        return TrackingOutput(frame_idx=0, tracks=[])

      def reset(self):
        self._frame_idx = 0
        self._trajectories.clear()

    tracker = MinimalTracker({})

    # Add trajectory points
    tracker._update_trajectory(1, np.array([10, 20]))
    tracker._update_trajectory(1, np.array([15, 25]))
    tracker._update_trajectory(2, np.array([100, 200]))

    # Test get_trajectories
    trajectories = tracker.get_trajectories()
    assert 1 in trajectories
    assert 2 in trajectories
    assert len(trajectories[1]) == 2

    # Test get_trajectory
    traj = tracker.get_trajectory(1)
    assert traj is not None
    assert len(traj) == 2
    assert np.allclose(traj[0], [10, 20])

    # Non-existent track
    assert tracker.get_trajectory(999) is None


class TestUltralyticsTracker:
  """Test the UltralyticsTracker wrapper."""

  def test_tracker_initialization(self):
    from src.tracking.ultralytics_tracker import UltralyticsTracker

    mock_ultralytics = MagicMock()
    mock_ultralytics.YOLO.return_value = MagicMock()

    with patch.dict("sys.modules", {"ultralytics": mock_ultralytics}):
      tracker = UltralyticsTracker(
        config=_tracker_config(),
        model_path="yolov8n.pt",
        device="cpu",
      )

    assert tracker.tracker_type == "botsort"
    assert tracker.device == "cpu"
    assert tracker.track_buffer == 30

  def test_update_requires_frame(self):
    from src.tracking.ultralytics_tracker import UltralyticsTracker

    mock_ultralytics = MagicMock()
    mock_ultralytics.YOLO.return_value = MagicMock()

    with patch.dict("sys.modules", {"ultralytics": mock_ultralytics}):
      tracker = UltralyticsTracker(_tracker_config(), device="cpu")

    detections = np.array([[0, 0, 50, 50]])
    confidences = np.array([0.9])
    class_ids = np.array([0])

    with pytest.raises(ValueError, match="requires frame"):
      tracker.update(detections, confidences, class_ids, frame=None)

  def test_update_returns_tracks(self):
    from src.tracking.ultralytics_tracker import UltralyticsTracker

    fake_yolo = MagicMock()
    fake_yolo.track.return_value = [_make_track_result(3)]

    mock_ultralytics = MagicMock()
    mock_ultralytics.YOLO.return_value = fake_yolo

    with patch.dict("sys.modules", {"ultralytics": mock_ultralytics}):
      tracker = UltralyticsTracker(_tracker_config(), device="cpu")

      # Create dummy frame
      frame = np.zeros((480, 640, 3), dtype=np.uint8)
      detections = np.array([[0, 0, 50, 50]])
      confidences = np.array([0.9])
      class_ids = np.array([0])

      output = tracker.update(detections, confidences, class_ids, frame)

    assert output.frame_idx == 1
    assert len(output.tracks) == 3
    assert output.tracks[0].track_id == 1
    assert output.tracks[1].track_id == 2
    assert output.tracks[2].track_id == 3

  def test_update_empty_result(self):
    from src.tracking.ultralytics_tracker import UltralyticsTracker

    fake_yolo = MagicMock()
    fake_yolo.track.return_value = [_make_empty_track_result()]

    mock_ultralytics = MagicMock()
    mock_ultralytics.YOLO.return_value = fake_yolo

    with patch.dict("sys.modules", {"ultralytics": mock_ultralytics}):
      tracker = UltralyticsTracker(_tracker_config(), device="cpu")
      frame = np.zeros((480, 640, 3), dtype=np.uint8)
      output = tracker.update(np.array([]), np.array([]), np.array([]), frame)

    assert len(output.tracks) == 0

  def test_movement_vector_computation(self):
    from src.tracking.ultralytics_tracker import UltralyticsTracker

    fake_yolo = MagicMock()

    # First frame: track at position (25, 25)
    result1 = _make_track_result(1, track_ids=[1])
    # Second frame: track moved to (75, 75) - so centroid moves from (25, 25) to (75, 75)
    result2 = MagicMock()
    result2.boxes = MagicMock()
    result2.boxes.xyxy.cpu().numpy.return_value = np.array(
      [[50, 50, 100, 100]], dtype=np.float32
    )
    result2.boxes.id = MagicMock()
    result2.boxes.id.cpu().numpy.return_value = np.array([1], dtype=np.float32)
    result2.boxes.conf.cpu().numpy.return_value = np.array([0.9], dtype=np.float32)
    result2.boxes.cls.cpu().numpy.return_value = np.array([0], dtype=np.float32)

    fake_yolo.track.side_effect = [[result1], [result2]]

    mock_ultralytics = MagicMock()
    mock_ultralytics.YOLO.return_value = fake_yolo

    with patch.dict("sys.modules", {"ultralytics": mock_ultralytics}):
      tracker = UltralyticsTracker(_tracker_config(), device="cpu")
      frame = np.zeros((480, 640, 3), dtype=np.uint8)

      # First update - no velocity yet
      output1 = tracker.update(np.array([]), np.array([]), np.array([]), frame)
      assert output1.tracks[0].speed == 0.0

      # Second update - should have velocity
      output2 = tracker.update(np.array([]), np.array([]), np.array([]), frame)
      track = output2.tracks[0]

      # Velocity should be (50, 50) - moving 50 pixels right and 50 down
      assert np.allclose(track.velocity, [50, 50], atol=1)
      assert track.speed > 0
      # Direction should be ~45 degrees (moving diagonally down-right)
      assert 40 <= track.direction <= 50

  def test_trajectory_accumulation(self):
    from src.tracking.ultralytics_tracker import UltralyticsTracker

    fake_yolo = MagicMock()
    fake_yolo.track.return_value = [_make_track_result(1, track_ids=[1])]

    mock_ultralytics = MagicMock()
    mock_ultralytics.YOLO.return_value = fake_yolo

    with patch.dict("sys.modules", {"ultralytics": mock_ultralytics}):
      tracker = UltralyticsTracker(_tracker_config(), device="cpu")
      frame = np.zeros((480, 640, 3), dtype=np.uint8)

      # Run several updates
      for _ in range(5):
        tracker.update(np.array([]), np.array([]), np.array([]), frame)

      trajectories = tracker.get_trajectories()
      assert 1 in trajectories
      assert len(trajectories[1]) == 5

  def test_reset_clears_state(self):
    from src.tracking.ultralytics_tracker import UltralyticsTracker

    fake_yolo = MagicMock()
    fake_yolo.track.return_value = [_make_track_result(1)]

    mock_ultralytics = MagicMock()
    mock_ultralytics.YOLO.return_value = fake_yolo

    with patch.dict("sys.modules", {"ultralytics": mock_ultralytics}):
      tracker = UltralyticsTracker(_tracker_config(), device="cpu")
      frame = np.zeros((480, 640, 3), dtype=np.uint8)

      tracker.update(np.array([]), np.array([]), np.array([]), frame)
      assert tracker.frame_idx == 1
      assert len(tracker.get_trajectories()) > 0

      tracker.reset()

      assert tracker.frame_idx == 0
      assert len(tracker.get_trajectories()) == 0


class TestCountingLine:
  """Test the CountingLine class."""

  def test_counting_line_creation(self):
    from src.tracking.counting import CountingLine

    line = CountingLine(
      line_id=1,
      start=[0, 100],
      end=[640, 100],
      name="entrance",
    )

    assert line.line_id == 1
    assert line.name == "entrance"
    assert np.allclose(line.start, [0, 100])
    assert np.allclose(line.end, [640, 100])

  def test_get_side_horizontal_line(self):
    from src.tracking.counting import CountingLine

    # Horizontal line at y=100
    line = CountingLine(1, [0, 100], [640, 100])

    # Point above line (y < 100)
    assert line.get_side(np.array([320, 50])) != 0

    # Point below line (y > 100)
    side_below = line.get_side(np.array([320, 150]))
    side_above = line.get_side(np.array([320, 50]))
    assert side_below != side_above  # Different sides

  def test_get_side_vertical_line(self):
    from src.tracking.counting import CountingLine

    # Vertical line at x=320
    line = CountingLine(1, [320, 0], [320, 480])

    # Point to the left
    side_left = line.get_side(np.array([100, 240]))
    # Point to the right
    side_right = line.get_side(np.array([500, 240]))

    assert side_left != side_right


class TestCountingZone:
  """Test the CountingZone class."""

  def test_counting_zone_creation(self):
    from src.tracking.counting import CountingZone

    vertices = [[0, 0], [100, 0], [100, 100], [0, 100]]
    zone = CountingZone(zone_id=1, vertices=vertices, name="roi")

    assert zone.zone_id == 1
    assert zone.name == "roi"
    assert zone.vertices.shape == (4, 2)

  def test_contains_inside_point(self):
    from src.tracking.counting import CountingZone

    # Square zone from (0,0) to (100,100)
    zone = CountingZone(1, [[0, 0], [100, 0], [100, 100], [0, 100]])

    # Point inside
    assert zone.contains(np.array([50, 50])) is True
    assert zone.contains(np.array([10, 10])) is True
    assert zone.contains(np.array([90, 90])) is True

  def test_contains_outside_point(self):
    from src.tracking.counting import CountingZone

    zone = CountingZone(1, [[0, 0], [100, 0], [100, 100], [0, 100]])

    # Points outside
    assert zone.contains(np.array([150, 50])) is False
    assert zone.contains(np.array([-10, 50])) is False
    assert zone.contains(np.array([50, 150])) is False


class TestVehicleCounter:
  """Test the VehicleCounter for redundant counting prevention."""

  def test_counter_initialization(self):
    from src.tracking.counting import VehicleCounter

    counter = VehicleCounter()
    assert repr(counter) == "VehicleCounter(lines=0, zones=0, unique_ids=0)"

  def test_add_line_and_zone(self):
    from src.tracking.counting import CountingLine, CountingZone, VehicleCounter

    counter = VehicleCounter()
    counter.add_line(CountingLine(1, [0, 100], [640, 100], "line1"))
    counter.add_zone(CountingZone(1, [[0, 0], [100, 0], [100, 100], [0, 100]], "zone1"))

    assert "lines=1" in repr(counter)
    assert "zones=1" in repr(counter)

  def test_line_crossing_counted_once(self):
    from src.tracking.base import Track
    from src.tracking.counting import CountingLine, VehicleCounter

    counter = VehicleCounter()
    # Horizontal line at y=100
    counter.add_line(CountingLine(1, [0, 100], [640, 100]))

    # Track starts above line (y=50), then crosses below (y=150)
    track_above = Track(
      track_id=1,
      bbox=np.array([300, 25, 350, 75]),  # centroid at (325, 50)
      confidence=0.9,
      class_id=0,
    )
    track_below = Track(
      track_id=1,
      bbox=np.array([300, 125, 350, 175]),  # centroid at (325, 150)
      confidence=0.9,
      class_id=0,
    )

    # First update - track above line
    result1 = counter.update([track_above])
    assert result1.line_crossings[1] == 0  # No crossing yet

    # Second update - track below line (crossed!)
    result2 = counter.update([track_below])
    assert result2.line_crossings[1] == 1

    # Third update - still below (no new crossing)
    result3 = counter.update([track_below])
    assert result3.line_crossings[1] == 1  # Still 1, not 2

    # Fourth update - back above (should NOT count again - already crossed this line)
    result4 = counter.update([track_above])
    assert result4.line_crossings[1] == 1  # Still 1

  def test_zone_entry_counted_once(self):
    from src.tracking.base import Track
    from src.tracking.counting import CountingZone, VehicleCounter

    counter = VehicleCounter()
    counter.add_zone(CountingZone(1, [[200, 200], [400, 200], [400, 400], [200, 400]]))

    # Track outside zone
    track_outside = Track(
      track_id=1,
      bbox=np.array([0, 0, 50, 50]),
      confidence=0.9,
      class_id=0,
    )
    # Track inside zone
    track_inside = Track(
      track_id=1,
      bbox=np.array([275, 275, 325, 325]),  # centroid at (300, 300)
      confidence=0.9,
      class_id=0,
    )

    # Outside - no entry
    result1 = counter.update([track_outside])
    assert result1.zone_entries[1] == 0

    # Inside - first entry
    result2 = counter.update([track_inside])
    assert result2.zone_entries[1] == 1

    # Still inside - no new entry
    result3 = counter.update([track_inside])
    assert result3.zone_entries[1] == 1

  def test_unique_id_tracking(self):
    from src.tracking.base import Track
    from src.tracking.counting import VehicleCounter

    counter = VehicleCounter()

    tracks = [
      Track(track_id=1, bbox=np.array([0, 0, 50, 50]), confidence=0.9, class_id=0),
      Track(
        track_id=2, bbox=np.array([100, 100, 150, 150]), confidence=0.9, class_id=0
      ),
      Track(
        track_id=3, bbox=np.array([200, 200, 250, 250]), confidence=0.9, class_id=0
      ),
    ]

    result = counter.update(tracks)
    assert result.total_unique_ids == 3

    # Update with same tracks
    result = counter.update(tracks)
    assert result.total_unique_ids == 3  # Still 3, not 6

    # Add new track
    tracks.append(
      Track(track_id=4, bbox=np.array([300, 300, 350, 350]), confidence=0.9, class_id=0)
    )
    result = counter.update(tracks)
    assert result.total_unique_ids == 4

  def test_reset_clears_all_state(self):
    from src.tracking.base import Track
    from src.tracking.counting import CountingLine, CountingZone, VehicleCounter

    counter = VehicleCounter()
    counter.add_line(CountingLine(1, [0, 100], [640, 100]))
    counter.add_zone(CountingZone(1, [[0, 0], [100, 0], [100, 100], [0, 100]]))

    track = Track(
      track_id=1,
      bbox=np.array([25, 25, 75, 75]),
      confidence=0.9,
      class_id=0,
    )
    counter.update([track])

    assert counter.get_counts().total_unique_ids == 1

    counter.reset()

    result = counter.get_counts()
    assert result.total_unique_ids == 0
    assert result.line_crossings[1] == 0
    assert result.zone_entries[1] == 0

  def test_crossing_direction_tracking(self):
    from src.tracking.base import Track
    from src.tracking.counting import CountingLine, VehicleCounter

    counter = VehicleCounter()
    counter.add_line(CountingLine(1, [0, 100], [640, 100]))

    # Track 1: crosses from above to below (positive)
    track1_above = Track(
      track_id=1, bbox=np.array([100, 25, 150, 75]), confidence=0.9, class_id=0
    )
    track1_below = Track(
      track_id=1, bbox=np.array([100, 125, 150, 175]), confidence=0.9, class_id=0
    )

    # Track 2: crosses from below to above (negative)
    track2_below = Track(
      track_id=2, bbox=np.array([300, 125, 350, 175]), confidence=0.9, class_id=0
    )
    track2_above = Track(
      track_id=2, bbox=np.array([300, 25, 350, 75]), confidence=0.9, class_id=0
    )

    counter.update([track1_above, track2_below])
    result = counter.update([track1_below, track2_above])

    assert result.line_crossings[1] == 2
    # One positive, one negative crossing
    assert result.positive_crossings + result.negative_crossings == 2

  def test_remove_stale_tracks(self):
    from src.tracking.base import Track
    from src.tracking.counting import VehicleCounter

    counter = VehicleCounter()

    tracks = [
      Track(track_id=1, bbox=np.array([0, 0, 50, 50]), confidence=0.9, class_id=0),
      Track(
        track_id=2, bbox=np.array([100, 100, 150, 150]), confidence=0.9, class_id=0
      ),
    ]
    counter.update(tracks)

    # Remove track 2 from active set
    counter.remove_stale_tracks({1})

    # Internal state for track 2 should be cleaned up
    # (can't directly test internal state, but ensures no error)
    result = counter.get_counts()
    assert result.total_unique_ids == 2  # Still counted historically
