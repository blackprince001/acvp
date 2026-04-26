"""Video and image sequence loading utilities for traffic scenes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import yaml
from loguru import logger

SUPPORTED_VIDEO_EXT = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
SUPPORTED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


@dataclass
class FrameMetadata:
  """Metadata for a single loaded frame."""

  frame_idx: int
  timestamp: float
  source_path: str
  scene_name: str
  original_width: int
  original_height: int
  fps: float = 30.0


@dataclass
class SceneInfo:
  """Metadata about a raw scene directory."""

  name: str
  scene_type: str = "unknown"
  location: str = "unknown"
  camera_height_m: float = 0.0
  camera_angle_deg: float = 0.0
  resolution: tuple[int, int] = (0, 0)
  fps: float = 30.0
  weather: str = "clear"
  duration_seconds: float = 0.0
  road_lanes: int = 0
  has_sidewalk: bool = False
  has_service_road: bool = False
  raw: dict = field(default_factory=dict)


class RawLoader:
  """Load video files or image sequences from raw scene directories.

  Supports MP4, AVI, MKV, MOV, WebM video formats and JPEG/PNG/BMP/TIFF
  image sequences.  Returns a frame iterator that yields ``(frame,
  metadata)`` tuples.
  """

  def __init__(
    self,
    scene_dir: str | Path,
    target_fps: int | None = None,
    max_frames: int | None = None,
  ) -> None:
    """Initialise the loader.

    Args:
        scene_dir: Path to ``data/raw/<scene_name>/``.
        target_fps: If set, skip frames to achieve this rate.
            ``None`` means use the source FPS.
        max_frames: Maximum number of frames to yield (useful for
            quick debugging).  ``None`` means no limit.
    """
    self.scene_dir = Path(scene_dir)
    self.target_fps = target_fps
    self.max_frames = max_frames

    if not self.scene_dir.is_dir():
      raise FileNotFoundError(f"Scene directory not found: {self.scene_dir}")

    self.scene_name = self.scene_dir.name
    self.scene_info = self._load_metadata()
    self._source_type = self._detect_source_type()

    logger.info(
      "RawLoader initialised for scene '{}' (type: {})",
      self.scene_name,
      self._source_type,
    )

  # ------------------------------------------------------------------
  # Public API
  # ------------------------------------------------------------------

  def frames(
    self,
  ) -> Generator[tuple[np.ndarray, FrameMetadata], None, None]:
    """Iterate over frames from the scene.

    Yields:
        ``(frame_image, metadata)`` where *frame_image* is a
        BGR ``np.ndarray`` from OpenCV.
    """
    if self._source_type == "video":
      yield from self._iter_video()
    elif self._source_type == "image_sequence":
      yield from self._iter_images()
    else:
      raise RuntimeError(f"No supported video or image files found in {self.scene_dir}")

  def get_scene_info(self) -> SceneInfo:
    """Return parsed scene metadata."""
    return self.scene_info

  def count_frames(self) -> int:
    """Return the total number of frames available."""
    if self._source_type == "video":
      return self._count_video_frames()
    if self._source_type == "image_sequence":
      return len(self._list_images())
    return 0

  # ------------------------------------------------------------------
  # Internal helpers
  # ------------------------------------------------------------------

  def _detect_source_type(self) -> str:
    """Return ``'video'``, ``'image_sequence'``, or ``'none'``."""
    if self._list_videos():
      return "video"
    if self._list_images():
      return "image_sequence"
    return "none"

  def _list_videos(self) -> list[Path]:
    return sorted(
      p for p in self.scene_dir.iterdir() if p.suffix.lower() in SUPPORTED_VIDEO_EXT
    )

  def _list_images(self) -> list[Path]:
    return sorted(
      p for p in self.scene_dir.iterdir() if p.suffix.lower() in SUPPORTED_IMAGE_EXT
    )

  def _load_metadata(self) -> SceneInfo:
    """Load ``metadata.yaml`` from the scene directory."""
    meta_path = self.scene_dir / "metadata.yaml"
    if not meta_path.exists():
      logger.warning("No metadata.yaml found in {}; using defaults", self.scene_dir)
      return SceneInfo(name=self.scene_name)

    with open(meta_path) as fh:
      raw: dict = yaml.safe_load(fh) or {}

    scene = raw.get("scene", {})
    camera = scene.get("camera", {})
    recording = scene.get("recording", {})
    road = scene.get("road", {})

    res = camera.get("resolution", [0, 0])
    info = SceneInfo(
      name=scene.get("name", self.scene_name),
      scene_type=scene.get("type", "unknown"),
      location=scene.get("location", "unknown"),
      camera_height_m=camera.get("height_m", 0.0),
      camera_angle_deg=camera.get("angle_deg", 0.0),
      resolution=(res[0], res[1]) if isinstance(res, (list, tuple)) else (0, 0),
      fps=camera.get("fps", 30.0),
      weather=recording.get("weather", "clear"),
      duration_seconds=recording.get("duration_seconds", 0.0),
      road_lanes=road.get("lanes", 0),
      has_sidewalk=road.get("has_sidewalk", False),
      has_service_road=road.get("has_service_road", False),
      raw=raw,
    )
    logger.debug("Loaded scene metadata: {}", info)
    return info

  # -- Video iteration ------------------------------------------------

  def _iter_video(
    self,
  ) -> Generator[tuple[np.ndarray, FrameMetadata], None, None]:
    videos = self._list_videos()
    if not videos:
      return

    video_path = videos[0]
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
      raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = self.target_fps if self.target_fps else source_fps
    frame_skip = max(1, int(source_fps / fps)) if fps else 1

    logger.info(
      "Video: {} | {}x{} @ {:.1f} fps | {} total frames | skip={}",
      video_path.name,
      width,
      height,
      source_fps,
      total_frames,
      frame_skip,
    )

    frame_idx = 0
    yielded = 0

    while True:
      ret, frame = cap.read()
      if not ret:
        break

      if frame_idx % frame_skip == 0:
        timestamp = frame_idx / source_fps if source_fps else 0.0
        meta = FrameMetadata(
          frame_idx=frame_idx,
          timestamp=timestamp,
          source_path=str(video_path),
          scene_name=self.scene_name,
          original_width=width,
          original_height=height,
          fps=source_fps,
        )
        yield frame, meta
        yielded += 1
        if self.max_frames and yielded >= self.max_frames:
          break

      frame_idx += 1

    cap.release()
    logger.info("Yielded {} frames from video", yielded)

  def _count_video_frames(self) -> int:
    videos = self._list_videos()
    if not videos:
      return 0
    cap = cv2.VideoCapture(str(videos[0]))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count

  # -- Image sequence iteration ---------------------------------------

  def _iter_images(
    self,
  ) -> Generator[tuple[np.ndarray, FrameMetadata], None, None]:
    images = self._list_images()
    if not images:
      return

    fps = self.target_fps if self.target_fps else 30.0
    yielded = 0
    step = max(1, len(images) // self.max_frames) if self.max_frames else 1

    logger.info(
      "Image sequence: {} files | step={}",
      len(images),
      step,
    )

    for idx, img_path in enumerate(images):
      if idx % step != 0:
        continue

      frame = cv2.imread(str(img_path))
      if frame is None:
        logger.warning("Could not read image {}; skipping", img_path)
        continue

      height, width = frame.shape[:2]
      meta = FrameMetadata(
        frame_idx=idx,
        timestamp=idx / fps,
        source_path=str(img_path),
        scene_name=self.scene_name,
        original_width=width,
        original_height=height,
        fps=fps,
      )
      yield frame, meta
      yielded += 1

    logger.info("Yielded {} frames from image sequence", yielded)
