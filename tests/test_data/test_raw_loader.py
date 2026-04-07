"""Unit tests for src/data/raw_loader.py."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from src.data.raw_loader import FrameMetadata, RawLoader, SceneInfo


@pytest.fixture()
def tmp_scene_video(tmp_path: Path) -> Path:
  """Create a temporary scene directory with a synthetic video."""
  scene_dir = tmp_path / "test_scene_video"
  scene_dir.mkdir()

  video_path = scene_dir / "video.mp4"
  fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
  for i in range(90):
    frame = np.full((480, 640, 3), (i * 2 % 256, 100, 150), dtype=np.uint8)
    out.write(frame)
  out.release()

  return scene_dir


@pytest.fixture()
def tmp_scene_images(tmp_path: Path) -> Path:
  """Create a temporary scene directory with image sequence."""
  scene_dir = tmp_path / "test_scene_images"
  scene_dir.mkdir()

  for i in range(50):
    frame = np.full((480, 640, 3), (i * 5 % 256, 200, 100), dtype=np.uint8)
    cv2.imwrite(str(scene_dir / f"frame_{i:04d}.jpg"), frame)

  return scene_dir


@pytest.fixture()
def tmp_scene_with_metadata(tmp_path: Path) -> Path:
  """Create a scene directory with metadata.yaml."""
  scene_dir = tmp_path / "highway_scene"
  scene_dir.mkdir()

  metadata = {
    "scene": {
      "name": "highway_downtown",
      "type": "highway",
      "location": "City, Country",
      "camera": {
        "height_m": 8.5,
        "angle_deg": 35,
        "resolution": [1920, 1080],
        "fps": 30,
      },
      "recording": {
        "date": "2024-06-15",
        "weather": "clear",
        "duration_seconds": 3600,
      },
      "road": {
        "lanes": 4,
        "has_sidewalk": True,
        "has_service_road": False,
      },
    },
  }
  with open(scene_dir / "metadata.yaml", "w") as fh:
    yaml.dump(metadata, fh)

  video_path = scene_dir / "video.mp4"
  fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
  for i in range(60):
    frame = np.full((480, 640, 3), (i * 4 % 256, 50, 200), dtype=np.uint8)
    out.write(frame)
  out.release()

  return scene_dir


@pytest.fixture()
def tmp_empty_scene(tmp_path: Path) -> Path:
  """Create an empty scene directory with no valid media."""
  scene_dir = tmp_path / "empty_scene"
  scene_dir.mkdir()
  (scene_dir / "notes.txt").write_text("nothing here")
  return scene_dir


class TestVideoLoading:
  def test_iter_video_yields_frames(self, tmp_scene_video: Path) -> None:
    loader = RawLoader(tmp_scene_video)
    frames = list(loader.frames())

    assert len(frames) == 90
    for frame, meta in frames:
      assert isinstance(frame, np.ndarray)
      assert frame.shape == (480, 640, 3)
      assert isinstance(meta, FrameMetadata)

  def test_frame_metadata_fields(self, tmp_scene_video: Path) -> None:
    loader = RawLoader(tmp_scene_video)
    _, meta = next(loader.frames())

    assert meta.scene_name == "test_scene_video"
    assert meta.original_width == 640
    assert meta.original_height == 480
    assert meta.fps == 30.0
    assert meta.frame_idx == 0
    assert meta.timestamp == 0.0

  def test_max_frames_limits_output(self, tmp_scene_video: Path) -> None:
    loader = RawLoader(tmp_scene_video, max_frames=10)
    frames = list(loader.frames())

    assert len(frames) == 10

  def test_target_fps_skips_frames(self, tmp_scene_video: Path) -> None:
    loader = RawLoader(tmp_scene_video, target_fps=10)
    frames = list(loader.frames())

    assert len(frames) == 30

  def test_count_video_frames(self, tmp_scene_video: Path) -> None:
    loader = RawLoader(tmp_scene_video)
    assert loader.count_frames() == 90

  def test_get_scene_info_defaults(self, tmp_scene_video: Path) -> None:
    loader = RawLoader(tmp_scene_video)
    info = loader.get_scene_info()

    assert isinstance(info, SceneInfo)
    assert info.name == "test_scene_video"
    assert info.scene_type == "unknown"


class TestImageSequenceLoading:
  def test_iter_images_yields_frames(self, tmp_scene_images: Path) -> None:
    loader = RawLoader(tmp_scene_images)
    frames = list(loader.frames())

    assert len(frames) == 50
    for frame, meta in frames:
      assert isinstance(frame, np.ndarray)
      assert frame.shape == (480, 640, 3)
      assert isinstance(meta, FrameMetadata)

  def test_max_frames_limits_images(self, tmp_scene_images: Path) -> None:
    loader = RawLoader(tmp_scene_images, max_frames=10)
    frames = list(loader.frames())

    assert len(frames) == 10

  def test_image_metadata_fields(self, tmp_scene_images: Path) -> None:
    loader = RawLoader(tmp_scene_images)
    _, meta = next(loader.frames())

    assert meta.scene_name == "test_scene_images"
    assert meta.original_width == 640
    assert meta.original_height == 480
    assert meta.frame_idx == 0


class TestMetadataLoading:
  def test_loads_metadata_yaml(self, tmp_scene_with_metadata: Path) -> None:
    loader = RawLoader(tmp_scene_with_metadata)
    info = loader.get_scene_info()

    assert info.name == "highway_downtown"
    assert info.scene_type == "highway"
    assert info.location == "City, Country"
    assert info.camera_height_m == 8.5
    assert info.camera_angle_deg == 35
    assert info.resolution == (1920, 1080)
    assert info.fps == 30
    assert info.weather == "clear"
    assert info.duration_seconds == 3600
    assert info.road_lanes == 4
    assert info.has_sidewalk is True
    assert info.has_service_road is False

  def test_missing_metadata_uses_defaults(self, tmp_scene_video: Path) -> None:
    loader = RawLoader(tmp_scene_video)
    info = loader.get_scene_info()

    assert info.scene_type == "unknown"
    assert info.location == "unknown"


class TestErrorHandling:
  def test_nonexistent_directory_raises(self) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
      RawLoader("/nonexistent/path/to/scene")

  def test_empty_scene_raises_runtime(self, tmp_empty_scene: Path) -> None:
    loader = RawLoader(tmp_empty_scene)
    with pytest.raises(RuntimeError, match="No supported"):
      list(loader.frames())

  def test_count_empty_scene_returns_zero(self, tmp_empty_scene: Path) -> None:
    loader = RawLoader(tmp_empty_scene)
    assert loader.count_frames() == 0
