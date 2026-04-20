"""Video input/output for the inference pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import cv2
import numpy as np
from loguru import logger

from src.utils.types import FilteredResult, PredictionResult, TrackingResult


class VideoReader:
  """Iterate frames from a video file or image directory."""

  def __init__(self, source: str | Path):
    self.source = Path(source)
    if not self.source.exists():
      raise FileNotFoundError(f"Video source not found: {self.source}")

    self._is_dir = self.source.is_dir()
    self._cap: cv2.VideoCapture | None = None
    self._images: list[Path] = []

    if self._is_dir:
      self._images = sorted(
        p
        for p in self.source.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
      )
      if not self._images:
        raise RuntimeError(f"No images found in {self.source}")
      sample = cv2.imread(str(self._images[0]))
      self._height, self._width = sample.shape[:2]
      self._fps = 30.0
      self._frame_count = len(self._images)
    else:
      self._cap = cv2.VideoCapture(str(self.source))
      if not self._cap.isOpened():
        raise RuntimeError(f"Could not open video: {self.source}")
      self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
      self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
      self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
      self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(
      "VideoReader: {} | {}x{} @ {:.1f} fps | {} frames",
      self.source.name,
      self._width,
      self._height,
      self._fps,
      self._frame_count,
    )

  @property
  def fps(self) -> float:
    return self._fps

  @property
  def width(self) -> int:
    return self._width

  @property
  def height(self) -> int:
    return self._height

  @property
  def frame_count(self) -> int:
    return self._frame_count

  def frames(self) -> Generator[tuple[int, float, np.ndarray], None, None]:
    """Yield ``(frame_idx, timestamp, frame)`` tuples."""
    if self._is_dir:
      for idx, img_path in enumerate(self._images):
        frame = cv2.imread(str(img_path))
        if frame is None:
          logger.warning("Skipping unreadable image {}", img_path)
          continue
        yield idx, idx / self._fps, frame
      return

    assert self._cap is not None
    idx = 0
    while True:
      ret, frame = self._cap.read()
      if not ret:
        break
      yield idx, idx / self._fps, frame
      idx += 1

  def close(self):
    if self._cap is not None:
      self._cap.release()

  def __enter__(self):
    return self

  def __exit__(self, *args):
    self.close()


class VideoWriter:
  """Write processed frames to an output video file."""

  def __init__(
    self,
    output_path: str | Path,
    width: int,
    height: int,
    fps: float = 30.0,
    codec: str = "mp4v",
  ):
    self.output_path = Path(output_path)
    self.output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    self._writer = cv2.VideoWriter(str(self.output_path), fourcc, fps, (width, height))
    if not self._writer.isOpened():
      raise RuntimeError(f"Could not open video writer for {self.output_path}")

    self._width = width
    self._height = height
    self._frame_count = 0
    logger.info(
      "VideoWriter: {} | {}x{} @ {:.1f} fps | codec={}",
      self.output_path.name,
      width,
      height,
      fps,
      codec,
    )

  def write(self, frame: np.ndarray):
    if frame.shape[1] != self._width or frame.shape[0] != self._height:
      frame = cv2.resize(frame, (self._width, self._height))
    self._writer.write(frame)
    self._frame_count += 1

  def close(self):
    self._writer.release()
    logger.info(
      "VideoWriter closed: {} frames written to {}", self._frame_count, self.output_path
    )

  def __enter__(self):
    return self

  def __exit__(self, *args):
    self.close()


def draw_overlay(
  frame: np.ndarray,
  tracking: TrackingResult | None = None,
  filtered: FilteredResult | None = None,
  prediction: PredictionResult | None = None,
  road_mask: np.ndarray | None = None,
) -> np.ndarray:
  """Render detections, tracks, road mask, and predictions onto the frame."""
  out = frame.copy()

  if road_mask is not None:
    mask = road_mask.astype(bool)
    if mask.shape[:2] == out.shape[:2]:
      overlay = out.copy()
      overlay[mask] = (0, 255, 0)
      out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)

  if tracking is not None:
    for obj in tracking.tracked_objects:
      box = obj.get("box")
      track_id = obj.get("track_id", -1)
      if box is None:
        continue
      x1, y1, x2, y2 = (int(v) for v in box[:4])
      cv2.rectangle(out, (x1, y1), (x2, y2), (255, 128, 0), 2)
      cv2.putText(
        out,
        f"id={track_id}",
        (x1, max(y1 - 6, 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 128, 0),
        1,
        cv2.LINE_AA,
      )

  y = 24
  if filtered is not None:
    _put_text(out, f"on-road: {filtered.vehicle_count}", (12, y))
    y += 24
    _put_text(out, f"occupancy: {filtered.occupancy_ratio:.3f}", (12, y))
    y += 24

  if prediction is not None:
    _put_text(out, f"density(now): {prediction.current_density:.2f}", (12, y))
    y += 24
    preds = prediction.predicted_density
    if preds is not None and preds.size > 0:
      _put_text(out, f"pred(T+1): {float(preds.flat[0]):.2f}", (12, y))

  return out


def _put_text(frame: np.ndarray, text: str, org: tuple[int, int]):
  cv2.putText(
    frame,
    text,
    org,
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (0, 0, 0),
    3,
    cv2.LINE_AA,
  )
  cv2.putText(
    frame,
    text,
    org,
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (255, 255, 255),
    1,
    cv2.LINE_AA,
  )
