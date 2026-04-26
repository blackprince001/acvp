"""Tests for feature engineering module."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from src.features.extractor import FEATURE_NAMES, N_FEATURES, FeatureExtractor
from src.features.normalizer import FeatureNormalizer
from src.features.windowing import WindowBuilder
from src.utils.types import FilteredResult


def _make_filtered(
  vehicles: list[dict], occupancy: float = 0.2, frame_idx: int = 0
) -> FilteredResult:
  return FilteredResult(
    frame_idx=frame_idx,
    timestamp=float(frame_idx),
    on_road_vehicles=vehicles,
    occupancy_ratio=occupancy,
    vehicle_count=len(vehicles),
  )


def _vehicle(track_id: int, speed: float, direction: float) -> dict:
  return {
    "track_id": track_id,
    "bbox": np.array([0.0, 0.0, 10.0, 10.0]),
    "centroid": np.array([5.0, 5.0]),
    "confidence": 0.9,
    "class_id": 0,
    "speed": speed,
    "direction": direction,
    "overlap": 0.8,
  }


def _road(h: int = 100, w: int = 100) -> np.ndarray:
  return np.ones((h, w), dtype=bool)


def test_feature_vector_length():
  fe = FeatureExtractor()
  f = fe.extract(_make_filtered([]), _road())
  assert f.shape == (N_FEATURES,)
  assert N_FEATURES == 10
  assert len(FEATURE_NAMES) == 10


def test_feature_dtype():
  fe = FeatureExtractor()
  f = fe.extract(_make_filtered([]), _road())
  assert f.dtype == np.float64


def test_vehicle_count_empty():
  fe = FeatureExtractor()
  f = fe.extract(_make_filtered([]), _road())
  assert f[0] == pytest.approx(0.0)


def test_vehicle_count_three():
  fe = FeatureExtractor()
  vehicles = [_vehicle(i, 5.0, 90.0) for i in range(3)]
  f = fe.extract(_make_filtered(vehicles), _road())
  assert f[0] == pytest.approx(3.0)


def test_occupancy_ratio_passthrough():
  fe = FeatureExtractor()
  f = fe.extract(_make_filtered([], occupancy=0.42), _road())
  assert f[1] == pytest.approx(0.42)


def test_mean_speed_empty():
  fe = FeatureExtractor()
  assert fe.extract(_make_filtered([]), _road())[2] == pytest.approx(0.0)


def test_mean_speed_values():
  fe = FeatureExtractor()
  vehicles = [_vehicle(1, 10.0, 0.0), _vehicle(2, 20.0, 0.0)]
  f = fe.extract(_make_filtered(vehicles), _road())
  assert f[2] == pytest.approx(15.0)


def test_mean_direction_empty():
  fe = FeatureExtractor()
  assert fe.extract(_make_filtered([]), _road())[3] == pytest.approx(0.0)


def test_mean_direction_values():
  fe = FeatureExtractor()
  vehicles = [_vehicle(1, 5.0, 30.0), _vehicle(2, 5.0, 90.0)]
  f = fe.extract(_make_filtered(vehicles), _road())
  assert f[3] == pytest.approx(60.0)


def test_density_zero_road_area():
  fe = FeatureExtractor()
  empty_road = np.zeros((100, 100), dtype=bool)
  vehicles = [_vehicle(1, 5.0, 0.0)]
  f = fe.extract(_make_filtered(vehicles), empty_road)
  assert f[4] == pytest.approx(0.0)


def test_density_value():
  fe = FeatureExtractor()
  road = np.ones((10, 10), dtype=bool)  # 100 pixels
  vehicles = [_vehicle(i, 5.0, 0.0) for i in range(5)]
  f = fe.extract(_make_filtered(vehicles), road)
  assert f[4] == pytest.approx(5 / 100)


def test_flow_passthrough():
  fe = FeatureExtractor()
  f = fe.extract(_make_filtered([]), _road(), flow=7.0)
  assert f[5] == pytest.approx(7.0)


def test_congestion_zero_when_no_vehicles():
  fe = FeatureExtractor()
  f = fe.extract(_make_filtered([], occupancy=0.5), _road())
  assert f[6] == pytest.approx(0.0)


def test_congestion_zero_when_at_max_speed():
  fe = FeatureExtractor(max_speed=10.0)
  vehicles = [_vehicle(1, 10.0, 0.0)]
  f = fe.extract(_make_filtered(vehicles, occupancy=1.0), _road())
  assert f[6] == pytest.approx(0.0)


def test_congestion_clamped_to_one():
  fe = FeatureExtractor(max_speed=10.0)
  vehicles = [_vehicle(1, 0.0, 0.0)]
  f = fe.extract(_make_filtered(vehicles, occupancy=1.0), _road())
  assert f[6] == pytest.approx(1.0)


def test_stopped_ratio_all_stopped():
  fe = FeatureExtractor(stop_threshold=2.0)
  vehicles = [_vehicle(i, 0.5, 0.0) for i in range(4)]
  f = fe.extract(_make_filtered(vehicles), _road())
  assert f[7] == pytest.approx(1.0)


def test_stopped_ratio_none_stopped():
  fe = FeatureExtractor(stop_threshold=1.0)
  vehicles = [_vehicle(i, 5.0, 0.0) for i in range(3)]
  f = fe.extract(_make_filtered(vehicles), _road())
  assert f[7] == pytest.approx(0.0)


def test_stopped_ratio_half():
  fe = FeatureExtractor(stop_threshold=2.0)
  vehicles = [_vehicle(1, 0.5, 0.0), _vehicle(2, 5.0, 0.0)]
  f = fe.extract(_make_filtered(vehicles), _road())
  assert f[7] == pytest.approx(0.5)


def test_variance_zero_single_vehicle():
  fe = FeatureExtractor()
  vehicles = [_vehicle(1, 5.0, 45.0)]
  f = fe.extract(_make_filtered(vehicles), _road())
  assert f[8] == pytest.approx(0.0)
  assert f[9] == pytest.approx(0.0)


def test_speed_variance_value():
  fe = FeatureExtractor()
  vehicles = [_vehicle(1, 0.0, 0.0), _vehicle(2, 10.0, 0.0)]
  f = fe.extract(_make_filtered(vehicles), _road())
  assert f[8] == pytest.approx(np.array([0.0, 10.0]).var())


def _seq(T: int, F: int = 10) -> np.ndarray:
  return np.random.default_rng(0).random((T, F))


def test_window_builder_output_shapes():
  wb = WindowBuilder(input_len=5, horizon=3, stride=1)
  X, y = wb.build(_seq(20))
  # N = 20 - 5 - 3 + 1 = 13
  assert X.shape == (13, 5, 10)
  assert y.shape == (13, 3, 10)


def test_window_builder_stride():
  wb = WindowBuilder(input_len=5, horizon=3, stride=2)
  X, y = wb.build(_seq(20))
  assert X.shape[0] == 7  # ceil((20-8+1)/2) = 7


def test_window_builder_too_short_raises():
  wb = WindowBuilder(input_len=10, horizon=5)
  with pytest.raises(ValueError):
    wb.build(_seq(14))  # need 15


def test_window_builder_invalid_params():
  with pytest.raises(ValueError):
    WindowBuilder(input_len=0, horizon=1)


def test_build_single_shape():
  wb = WindowBuilder(input_len=5, horizon=3)
  seq = _seq(20)
  wf = wb.build_single(seq, end_idx=15)
  assert wf.features.shape == (5, 10)
  assert wf.window_start == 10
  assert wf.window_end == 15


def test_build_single_not_enough_history():
  wb = WindowBuilder(input_len=10, horizon=3)
  with pytest.raises(ValueError):
    wb.build_single(_seq(20), end_idx=5)


def _data(N: int = 100, F: int = 10) -> np.ndarray:
  rng = np.random.default_rng(42)
  return rng.random((N, F)) * 10 + 5  # values in [5, 15]


@pytest.mark.parametrize("method", ["standard", "minmax", "robust"])
def test_normalizer_round_trip(method):
  norm = FeatureNormalizer(method=method)
  X = _data()
  X_norm = norm.fit_transform(X)
  X_back = norm.inverse_transform(X_norm)
  np.testing.assert_allclose(X_back, X, atol=1e-10)


def test_normalizer_standard_zero_mean():
  norm = FeatureNormalizer("standard")
  X = _data()
  X_norm = norm.fit_transform(X)
  np.testing.assert_allclose(X_norm.mean(axis=0), 0.0, atol=1e-10)


def test_normalizer_minmax_range():
  norm = FeatureNormalizer("minmax")
  X = _data()
  X_norm = norm.fit_transform(X)
  assert float(X_norm.min()) >= 0.0 - 1e-10
  assert float(X_norm.max()) <= 1.0 + 1e-10


def test_normalizer_not_fitted_raises():
  norm = FeatureNormalizer()
  with pytest.raises(RuntimeError):
    norm.transform(_data())


def test_normalizer_invalid_method():
  with pytest.raises(ValueError):
    FeatureNormalizer(method="unknown")


def test_normalizer_constant_feature_no_div_zero():
  norm = FeatureNormalizer("standard")
  X = np.ones((10, 3))
  X_norm = norm.fit_transform(X)
  assert np.isfinite(X_norm).all()
