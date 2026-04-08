"""Unit tests for src/data/preprocess.py."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data.preprocess import (
  FramePreprocessor,
  PreprocessingConfig,
  iter_processed_frames,
  load_index,
  load_metadata,
)


@pytest.fixture()
def tmp_video_scene(tmp_path: Path) -> Path:
  scene_dir = tmp_path / "video_scene"
  scene_dir.mkdir()
  fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  out = cv2.VideoWriter(str(scene_dir / "video.mp4"), fourcc, 30.0, (640, 480))
  for i in range(60):
    frame = np.full((480, 640, 3), (i * 4 % 256, 100, 150), dtype=np.uint8)
    out.write(frame)
  out.release()
  return scene_dir


@pytest.fixture()
def tmp_output_dir(tmp_path: Path) -> Path:
  out = tmp_path / "processed"
  out.mkdir()
  return out


class TestFramePreprocessor:
  def test_process_video_creates_frames(
    self, tmp_video_scene: Path, tmp_output_dir: Path
  ) -> None:
    config = PreprocessingConfig(target_size=(320, 240))
    preprocessor = FramePreprocessor(config)
    result = preprocessor.process(tmp_video_scene, tmp_output_dir)

    assert result["frame_count"] == 60
    assert (tmp_output_dir / "frames").exists()
    assert (tmp_output_dir / "index.json").exists()
    assert (tmp_output_dir / "metadata.yaml").exists()

  def test_process_creates_valid_index(
    self, tmp_video_scene: Path, tmp_output_dir: Path
  ) -> None:
    preprocessor = FramePreprocessor()
    preprocessor.process(tmp_video_scene, tmp_output_dir)

    index = load_index(tmp_output_dir / "index.json")
    assert len(index) == 60
    assert "frame_idx" in index[0]
    assert "filename" in index[0]
    assert "timestamp" in index[0]

  def test_process_creates_metadata(
    self, tmp_video_scene: Path, tmp_output_dir: Path
  ) -> None:
    preprocessor = FramePreprocessor()
    preprocessor.process(tmp_video_scene, tmp_output_dir)

    meta = load_metadata(tmp_output_dir / "metadata.yaml")
    assert meta["frame_count"] == 60
    assert meta["target_size"] == [640, 640]
    assert meta["normalize"] is True

  def test_target_fps_limits_frames(
    self, tmp_video_scene: Path, tmp_output_dir: Path
  ) -> None:
    config = PreprocessingConfig(target_fps=10)
    preprocessor = FramePreprocessor(config)
    result = preprocessor.process(tmp_video_scene, tmp_output_dir)

    assert result["frame_count"] == 20

  def test_resize_to_custom_size(
    self, tmp_video_scene: Path, tmp_output_dir: Path
  ) -> None:
    config = PreprocessingConfig(target_size=(320, 320))
    preprocessor = FramePreprocessor(config)
    preprocessor.process(tmp_video_scene, tmp_output_dir)

    index = load_index(tmp_output_dir / "index.json")
    frame = cv2.imread(str(tmp_output_dir / "frames" / index[0]["filename"]))
    assert frame.shape[:2] == (320, 320)

  def test_normalize_flag_false(
    self, tmp_video_scene: Path, tmp_output_dir: Path
  ) -> None:
    config = PreprocessingConfig(normalize=False)
    preprocessor = FramePreprocessor(config)
    preprocessor.process(tmp_video_scene, tmp_output_dir)

    meta = load_metadata(tmp_output_dir / "metadata.yaml")
    assert meta["normalize"] is False


class TestHelperFunctions:
  def test_iter_processed_frames(
    self, tmp_video_scene: Path, tmp_output_dir: Path
  ) -> None:
    preprocessor = FramePreprocessor()
    preprocessor.process(tmp_video_scene, tmp_output_dir)

    frames = list(iter_processed_frames(tmp_output_dir))
    assert len(frames) == 60

    frame, info = frames[0]
    assert isinstance(frame, np.ndarray)
    assert "frame_idx" in info
    assert "timestamp" in info


class TestErrorHandling:
  def test_process_empty_directory_raises(self, tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    preprocessor = FramePreprocessor()

    with pytest.raises(ValueError, match="No supported media"):
      preprocessor.process(empty_dir, tmp_path / "output")
