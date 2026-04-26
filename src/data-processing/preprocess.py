"""Frame preprocessing pipeline for traffic video data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import yaml
from loguru import logger

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class PreprocessingConfig:
  """Configuration for frame preprocessing."""

  target_size: tuple[int, int] = (640, 640)
  target_fps: int | None = None
  normalize: bool = True
  keep_aspect_ratio: bool = False


class FramePreprocessor:
  """Preprocess video frames: resize, normalize, extract at target FPS.

  Output structure:
      output_dir/
          frames/
              frame_000001.jpg
              frame_000002.jpg
              ...
          index.json
          metadata.yaml
  """

  def __init__(self, config: PreprocessingConfig | None = None) -> None:
    self.config = config or PreprocessingConfig()

  def process(
    self,
    input_path: str | Path,
    output_dir: str | Path,
    progress: bool = True,
  ) -> dict:
    """Extract and preprocess frames from video or image sequence.

    Args:
        input_path: Path to video file or directory of images.
        output_dir: Where to write processed frames.
        progress: Show progress bar during processing.

    Returns:
        dict with keys: frame_count, output_dir, index_path,
                      metadata_path
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    sources = self._list_sources(input_path)
    if not sources:
      raise ValueError(f"No supported media found in {input_path}")

    all_frames = self._collect_frames(sources)
    fps = self._detect_fps(sources)
    frame_skip = self._calc_frame_skip(fps)

    logger.info(
      "Processing {} frames from {} sources, target FPS: {}, skip: {}",
      len(all_frames),
      len(sources),
      self.config.target_fps or fps,
      frame_skip,
    )

    index_data = []
    frame_idx = 0

    if sources and sources[0].suffix.lower() in {
      ".jpg",
      ".jpeg",
      ".png",
      ".bmp",
      ".tiff",
    }:
      for img_path in sources:
        frame = cv2.imread(str(img_path))
        if frame is None:
          logger.warning("Could not read image {}; skipping", img_path)
          continue

        if frame_idx % frame_skip != 0:
          frame_idx += 1
          continue

        processed = self._preprocess_frame(frame)
        frame_name = f"frame_{frame_idx:06d}.jpg"
        out_path = frames_dir / frame_name
        cv2.imwrite(str(out_path), processed)

        index_data.append(
          {
            "frame_idx": frame_idx,
            "source_idx": 0,
            "filename": frame_name,
            "timestamp": frame_idx / fps,
          }
        )

        frame_idx += 1

        if progress and frame_idx % 500 == 0:
          logger.info("Processed {} frames...", frame_idx)

    else:
      for src_idx, src_path in enumerate(sources):
        cap = (
          cv2.VideoCapture(str(src_path))
          if src_path.suffix.lower() in {".mp4", ".avi", ".mkv", ".mov"}
          else None
        )

        try:
          while True:
            ret = False
            frame = None
            if cap:
              ret, frame = cap.read()
            elif src_path.is_dir():
              break

            if not ret:
              break

            if frame_idx % frame_skip != 0:
              frame_idx += 1
              continue

            processed = self._preprocess_frame(frame)
            frame_name = f"frame_{frame_idx:06d}.jpg"
            out_path = frames_dir / frame_name
            cv2.imwrite(str(out_path), processed)

            index_data.append(
              {
                "frame_idx": frame_idx,
                "source_idx": src_idx,
                "filename": frame_name,
                "timestamp": frame_idx / fps,
              }
            )

            frame_idx += 1

            if progress and frame_idx % 100 == 0:
              logger.info(f"Processed {frame_idx} frames...")

        finally:
          if cap:
            cap.release()

    index_path = output_dir / "index.json"
    with open(index_path, "w") as fh:
      json.dump(index_data, fh, indent=2)

    metadata = {
      "frame_count": len(index_data),
      "source_fps": fps,
      "target_fps": self.config.target_fps,
      "target_size": list(self.config.target_size),
      "normalize": self.config.normalize,
    }
    meta_path = output_dir / "metadata.yaml"
    with open(meta_path, "w") as fh:
      yaml.dump(metadata, fh)

    logger.info(
      "Preprocessing complete: {} frames -> {}",
      len(index_data),
      output_dir,
    )

    return {
      "frame_count": len(index_data),
      "output_dir": str(output_dir),
      "index_path": str(index_path),
      "metadata_path": str(meta_path),
    }

  def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
    """Resize and optionally normalize a single frame."""
    h, w = frame.shape[:2]
    target_w, target_h = self.config.target_size

    if self.config.keep_aspect_ratio:
      scale = min(target_w / w, target_h / h)
      new_w, new_h = int(w * scale), int(h * scale)
      resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

      processed = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
      y_offset = (target_h - new_h) // 2
      x_offset = (target_w - new_w) // 2
      processed[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
    else:
      processed = cv2.resize(
        frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR
      )

    if self.config.normalize:
      processed = processed.astype(np.float32) / 255.0

    return processed

  def _list_sources(self, input_path: Path) -> list[Path]:
    if input_path.is_file():
      return [input_path]
    if input_path.is_dir():
      videos = sorted(
        p
        for p in input_path.iterdir()
        if p.suffix.lower() in {".mp4", ".avi", ".mkv", ".mov"}
      )
      if videos:
        return videos
      return sorted(
        p
        for p in input_path.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
      )
    return []

  def _collect_frames(self, sources: list[Path]) -> list[Path]:
    if not sources:
      return []

    if sources[0].is_dir():
      frames = []
      for src in sources:
        frames.extend(
          sorted(
            p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
          )
        )
      return frames

    return sources

  def _detect_fps(self, sources: list[Path]) -> float:
    if not sources:
      return 30.0
    if sources[0].suffix.lower() in {".mp4", ".avi", ".mkv", ".mov"}:
      cap = cv2.VideoCapture(str(sources[0]))
      fps = cap.get(cv2.CAP_PROP_FPS)
      cap.release()
      return fps if fps > 0 else 30.0
    return 30.0

  def _calc_frame_skip(self, fps: float) -> int:
    if not self.config.target_fps:
      return 1
    return max(1, int(fps / self.config.target_fps))


def load_index(index_path: str | Path) -> list[dict]:
  """Load frame index file."""
  with open(index_path) as fh:
    return json.load(fh)


def load_metadata(meta_path: str | Path) -> dict:
  """Load preprocessing metadata."""
  with open(meta_path) as fh:
    return yaml.safe_load(fh)


def iter_processed_frames(
  processed_dir: str | Path,
) -> Generator[tuple[np.ndarray, dict], None, None]:
  """Iterate over preprocessed frames.

  Args:
      processed_dir: Path to output directory from process().

  Yields:
      (frame, info) where info contains frame_idx, timestamp, filename.
  """
  processed_dir = Path(processed_dir)
  index = load_index(processed_dir / "index.json")
  frames_dir = processed_dir / "frames"

  for item in index:
    frame = cv2.imread(str(frames_dir / item["filename"]))
    if frame is not None:
      yield frame, item
