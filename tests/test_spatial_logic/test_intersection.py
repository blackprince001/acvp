"""Tests for src/spatial_logic/intersection.py (T-055)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spatial_logic.intersection import compute_box_road_iou, compute_mask_road_iou


# ---------------------------------------------------------------------------
# compute_mask_road_iou
# ---------------------------------------------------------------------------

def test_mask_iou_full_overlap():
    m = np.ones((10, 10), dtype=bool)
    assert compute_mask_road_iou(m, m) == pytest.approx(1.0)


def test_mask_iou_no_overlap():
    v = np.zeros((10, 10), dtype=bool)
    v[:5, :] = True
    r = np.zeros((10, 10), dtype=bool)
    r[5:, :] = True
    assert compute_mask_road_iou(v, r) == pytest.approx(0.0)


def test_mask_iou_partial_overlap():
    v = np.zeros((10, 10), dtype=bool)
    v[:, :5] = True  # left half
    r = np.zeros((10, 10), dtype=bool)
    r[:, 3:] = True  # right 7 cols
    # intersection = cols 3-4 → 2 cols × 10 rows = 20
    # union = 10 cols × 10 rows = 100
    assert compute_mask_road_iou(v, r) == pytest.approx(20 / 100)


def test_mask_iou_both_empty():
    z = np.zeros((10, 10), dtype=bool)
    assert compute_mask_road_iou(z, z) == pytest.approx(0.0)


def test_mask_iou_accepts_uint8():
    v = np.ones((5, 5), dtype=np.uint8)
    r = np.ones((5, 5), dtype=np.uint8)
    assert compute_mask_road_iou(v, r) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_box_road_iou
# ---------------------------------------------------------------------------

def _full_road(h=100, w=100):
    return np.ones((h, w), dtype=bool)


def _empty_road(h=100, w=100):
    return np.zeros((h, w), dtype=bool)


def test_box_iou_full_road():
    road = _full_road()
    assert compute_box_road_iou([10, 10, 50, 50], road) == pytest.approx(1.0)


def test_box_iou_empty_road():
    road = _empty_road()
    assert compute_box_road_iou([10, 10, 50, 50], road) == pytest.approx(0.0)


def test_box_iou_half_road():
    road = np.zeros((100, 100), dtype=bool)
    road[:, 50:] = True  # right half
    # box spans full width [0, 0, 100, 100] → 50% on road
    assert compute_box_road_iou([0, 0, 100, 100], road) == pytest.approx(0.5)


def test_box_iou_zero_area_box():
    road = _full_road()
    assert compute_box_road_iou([10, 10, 10, 10], road) == pytest.approx(0.0)


def test_box_iou_out_of_bounds_clamped():
    road = _full_road(50, 50)
    # box extends beyond image — should clamp and not raise
    result = compute_box_road_iou([-10, -10, 200, 200], road)
    assert result == pytest.approx(1.0)


def test_box_iou_returns_float():
    road = _full_road()
    result = compute_box_road_iou([0, 0, 10, 10], road)
    assert isinstance(result, float)
