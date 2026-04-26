"""Tests for src/spatial_logic/filtering.py and occupancy.py."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from src.spatial_logic.filtering import OnRoadFilter
from src.spatial_logic.occupancy import compute_occupancy


@dataclass
class _FakeTrack:
  track_id: int
  bbox: np.ndarray
  confidence: float = 0.9
  class_id: int = 0
  centroid: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
  velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
  direction: float = 0.0
  speed: float = 5.0
  mask: np.ndarray | None = None


@dataclass
class _FakeTrackingOutput:
  tracks: list[_FakeTrack]
  frame_idx: int = 0


def _road_mask(h=100, w=100, road_cols=(0, 60)):
  """Road occupies columns road_cols[0]:road_cols[1]."""
  m = np.zeros((h, w), dtype=bool)
  m[:, road_cols[0] : road_cols[1]] = True
  return m


def test_filter_all_on_road():
  road = _road_mask()  # cols 0-60 are road
  tracks = [
    _FakeTrack(1, np.array([5, 5, 55, 55])),  # fully on road
    _FakeTrack(2, np.array([10, 10, 50, 50])),  # fully on road
  ]
  output = _FakeTrackingOutput(tracks)
  result = OnRoadFilter(overlap_threshold=0.5).filter(output, road)
  assert result.vehicle_count == 2
  assert len(result.on_road_vehicles) == 2


def test_filter_all_off_road():
  road = _road_mask()  # cols 0-60
  tracks = [
    _FakeTrack(1, np.array([70, 5, 95, 55])),  # fully off road (cols 70-95)
  ]
  output = _FakeTrackingOutput(tracks)
  result = OnRoadFilter(overlap_threshold=0.5).filter(output, road)
  assert result.vehicle_count == 0


def test_filter_mixed():
  road = _road_mask()
  tracks = [
    _FakeTrack(1, np.array([5, 5, 55, 55])),  # on road
    _FakeTrack(2, np.array([70, 5, 95, 55])),  # off road
  ]
  output = _FakeTrackingOutput(tracks)
  result = OnRoadFilter(overlap_threshold=0.5).filter(output, road)
  assert result.vehicle_count == 1
  assert result.on_road_vehicles[0]["track_id"] == 1


def test_filter_speed_gate_excludes_stationary():
  road = _road_mask()
  tracks = [
    _FakeTrack(1, np.array([5, 5, 55, 55]), speed=0.5),  # on road but slow
  ]
  output = _FakeTrackingOutput(tracks)
  result = OnRoadFilter(overlap_threshold=0.3, min_speed=1.0).filter(output, road)
  assert result.vehicle_count == 0


def test_filter_speed_gate_passes_fast():
  road = _road_mask()
  tracks = [
    _FakeTrack(1, np.array([5, 5, 55, 55]), speed=3.0),
  ]
  output = _FakeTrackingOutput(tracks)
  result = OnRoadFilter(overlap_threshold=0.3, min_speed=1.0).filter(output, road)
  assert result.vehicle_count == 1


def test_filter_uses_mask_when_available():
  road = np.zeros((100, 100), dtype=bool)
  road[0:50, 0:50] = True  # top-left quadrant

  vehicle_mask = np.zeros((100, 100), dtype=bool)
  vehicle_mask[0:50, 0:50] = True  # fully on road

  tracks = [_FakeTrack(1, np.array([0, 0, 50, 50]), mask=vehicle_mask)]
  output = _FakeTrackingOutput(tracks)
  result = OnRoadFilter(overlap_threshold=0.5).filter(output, road)
  assert result.vehicle_count == 1


def test_filter_result_fields():
  road = _road_mask()
  tracks = [_FakeTrack(1, np.array([5, 5, 55, 55]))]
  output = _FakeTrackingOutput(tracks)
  result = OnRoadFilter().filter(output, road, frame_idx=3, timestamp=1.5)
  assert result.frame_idx == 3
  assert result.timestamp == pytest.approx(1.5)
  assert 0.0 <= result.occupancy_ratio <= 1.0


def test_filter_empty_tracks():
  road = _road_mask()
  output = _FakeTrackingOutput(tracks=[])
  result = OnRoadFilter().filter(output, road)
  assert result.vehicle_count == 0
  assert result.occupancy_ratio == pytest.approx(0.0)


def test_occupancy_full_coverage():
  road = np.ones((10, 10), dtype=bool)
  vehicle = [np.ones((10, 10), dtype=bool)]
  assert compute_occupancy(vehicle, road) == pytest.approx(1.0)


def test_occupancy_no_vehicles():
  road = np.ones((10, 10), dtype=bool)
  assert compute_occupancy([], road) == pytest.approx(0.0)


def test_occupancy_empty_road():
  road = np.zeros((10, 10), dtype=bool)
  vehicle = [np.ones((10, 10), dtype=bool)]
  assert compute_occupancy(vehicle, road) == pytest.approx(0.0)


def test_occupancy_partial():
  road = np.zeros((10, 10), dtype=bool)
  road[:, :5] = True  # left half = 50 pixels
  vehicle = np.zeros((10, 10), dtype=bool)
  vehicle[:, :2] = True  # 20 pixels, all on road
  assert compute_occupancy([vehicle], road) == pytest.approx(20 / 50)


def test_occupancy_union_of_multiple_masks():
  road = np.ones((10, 10), dtype=bool)  # 100 pixels
  m1 = np.zeros((10, 10), dtype=bool)
  m1[:, :5] = True  # 50 px
  m2 = np.zeros((10, 10), dtype=bool)
  m2[:, 3:8] = True  # 50 px, overlaps m1 at cols 3-4
  # union = cols 0-7 = 80 px
  assert compute_occupancy([m1, m2], road) == pytest.approx(80 / 100)
