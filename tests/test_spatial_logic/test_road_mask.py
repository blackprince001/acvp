"""Tests for src/spatial_logic/road_mask.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from src.spatial_logic.road_mask import RoadMaskManager


@pytest.fixture()
def manager():
  return RoadMaskManager(default_shape=(100, 100))


@pytest.fixture()
def simple_mask():
  m = np.zeros((100, 100), dtype=np.uint8)
  m[20:80, 20:80] = 1
  return m


def test_load_static_npy(tmp_path, manager, simple_mask):
  p = tmp_path / "mask.npy"
  np.save(p, simple_mask)
  loaded = manager.load_static(p)
  assert loaded.dtype == bool
  assert loaded.shape == (100, 100)
  assert loaded[50, 50] is np.bool_(True)
  assert loaded[0, 0] is np.bool_(False)


def test_load_static_image(tmp_path, manager, simple_mask):
  pytest.importorskip("PIL")
  from PIL import Image

  p = tmp_path / "mask.png"
  Image.fromarray(simple_mask * 255).save(p)
  loaded = manager.load_static(p)
  assert loaded.dtype == bool
  assert loaded[50, 50] is np.bool_(True)


def test_set_and_get_static(manager, simple_mask):
  manager.set_static(simple_mask)
  mask = manager.get_mask()
  assert mask.dtype == bool
  np.testing.assert_array_equal(mask, simple_mask.astype(bool))


def test_get_mask_returns_static_when_no_per_frame(manager, simple_mask):
  manager.set_static(simple_mask)
  assert manager.get_mask(frame_id=42).sum() == simple_mask.sum()


def test_get_mask_default_all_true(manager):
  # No mask loaded — default_shape set → all-True mask
  mask = manager.get_mask()
  assert mask.shape == (100, 100)
  assert mask.all()


def test_get_mask_raises_without_mask_or_default():
  mgr = RoadMaskManager()  # no default_shape
  with pytest.raises(RuntimeError):
    mgr.get_mask()


def test_per_frame_overrides_static(manager, simple_mask):
  manager.set_static(simple_mask)
  per_frame = np.zeros((100, 100), dtype=bool)
  per_frame[0:10, 0:10] = True
  manager.set_frame_mask(5, per_frame)

  assert manager.get_mask(frame_id=5).sum() == 100  # 10×10
  assert manager.get_mask(frame_id=6).sum() == simple_mask.sum()  # static


def test_generate_from_segmentation_merges(manager):
  m1 = np.zeros((50, 50), dtype=bool)
  m1[0:25, :] = True
  m2 = np.zeros((50, 50), dtype=bool)
  m2[25:, :] = True

  merged = manager.generate_from_segmentation([m1, m2])
  assert merged.all()  # full coverage


def test_generate_stores_as_per_frame(manager):
  m = np.ones((50, 50), dtype=bool)
  manager.generate_from_segmentation([m], frame_id=7)
  assert manager.get_mask(frame_id=7).all()


def test_generate_stores_as_static_when_no_frame_id(manager):
  m = np.ones((50, 50), dtype=bool)
  manager.generate_from_segmentation([m])
  assert manager.get_mask().all()


def test_generate_raises_on_empty_list(manager):
  with pytest.raises(ValueError):
    manager.generate_from_segmentation([])
