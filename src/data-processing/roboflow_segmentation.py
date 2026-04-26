"""Roboflow API segmentation pipeline for traffic scenes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import yaml
from loguru import logger

try:
  import roboflow

  ROBOFLOW_AVAILABLE = True
except ImportError:
  ROBOFLOW_AVAILABLE = False


CLASS_ID_ROAD = 0
CLASS_ID_VEHICLE = 1


@dataclass
class RoboflowConfig:
  """Configuration for Roboflow segmentation."""

  api_key: str = ""
  workspace: str = ""
  project: str = ""
  model_version: str = "latest"
  batch_size: int = 32
  confidence_threshold: float = 0.25


class RoboflowSegmentationPipeline:
  """Send frames to Roboflow API for segmentation and download masks.

  Output structure:
      output_dir/
          masks/
              frame_000001.png
              frame_000002.png
              ...
          road_mask.png          # Static merged road mask
          annotations.yaml       # Class distributions
  """

  def __init__(self, config: RoboflowConfig) -> None:
    if not ROBOFLOW_AVAILABLE:
      raise ImportError("roboflow package is required: pip install roboflow")

    self.config = config
    self._client = None

  def segment_scene(
    self,
    frames_dir: str | Path,
    output_dir: str | Path,
    progress: bool = True,
  ) -> dict:
    """Run segmentation on all frames in a directory.

    Args:
        frames_dir: Path to preprocessed frames (from preprocess.py).
        output_dir: Where to save masks and annotations.
        progress: Show progress bar.

    Returns:
        dict with keys: frame_count, masks_dir, road_mask_path,
                      annotations_path
    """
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(
      frames_dir.glob("*.jpg") | frames_dir.glob("*.jpeg") | frames_dir.glob("*.png")
    )
    if not frame_files:
      raise ValueError(f"No frames found in {frames_dir}")

    logger.info("Segmenting {} frames via Roboflow...", len(frame_files))

    self._ensure_client()

    road_masks_accum = []
    vehicle_counts = []
    class_dist = {"road": 0, "vehicle": 0}

    for idx, frame_path in enumerate(frame_files):
      mask = self._segment_single(frame_path)

      if mask is not None:
        mask_path = masks_dir / frame_path.name
        cv2.imwrite(str(mask_path), mask)

        road_mask = (mask == CLASS_ID_ROAD).astype(np.uint8)
        vehicle_mask = (mask == CLASS_ID_VEHICLE).astype(np.uint8)

        road_masks_accum.append(road_mask)
        vehicle_counts.append(vehicle_mask.sum())
        class_dist["road"] += road_mask.sum()
        class_dist["vehicle"] += vehicle_mask.sum()

      if progress and (idx + 1) % 100 == 0:
        logger.info(f"Segmented {idx + 1}/{len(frame_files)} frames...")

    road_mask_static = self._generate_static_road_mask(road_masks_accum)
    road_mask_path = output_dir / "road_mask.png"
    cv2.imwrite(str(road_mask_path), (road_mask_static * 255).astype(np.uint8))

    annotations = {
      "frame_count": len(frame_files),
      "class_distribution": {
        "road_pixels": int(class_dist["road"]),
        "vehicle_pixels": int(class_dist["vehicle"]),
      },
      "avg_vehicle_pixels_per_frame": (
        int(np.mean(vehicle_counts)) if vehicle_counts else 0
      ),
    }
    anno_path = output_dir / "annotations.yaml"
    with open(anno_path, "w") as fh:
      yaml.dump(annotations, fh)

    logger.info(
      "Segmentation complete: {} frames -> {}",
      len(frame_files),
      output_dir,
    )

    return {
      "frame_count": len(frame_files),
      "masks_dir": str(masks_dir),
      "road_mask_path": str(road_mask_path),
      "annotations_path": str(anno_path),
    }

  def _ensure_client(self) -> None:
    if self._client is None:
      rf = roboflow.Roboflow(api_key=self.config.api_key)
      workspace = rf.workspace()
      project = workspace.project(self.config.project)
      self._client = project.version(self.config.model_version)
      logger.info(
        "Connected to Roboflow project: {}/{}",
        self.config.workspace,
        self.config.project,
      )

  def _segment_single(self, frame_path: Path) -> np.ndarray | None:
    try:
      response = self._client.predict(str(frame_path), hosted=True)

      h, w = 640, 640
      mask = np.zeros((h, w), dtype=np.uint8)

      for pred in response:
        if hasattr(pred, "class_id"):
          class_id = pred.class_id
          points = pred.points
        elif isinstance(pred, dict):
          class_id = pred.get("class", pred.get("class_id", 1))
          points = pred.get("points", [])
        else:
          continue

        class_id = int(class_id)

        if class_id in [0, 1]:
          poly = np.array([[int(p["x"]), int(p["y"])] for p in points], np.int32)
          cv2.fillPoly(mask, [poly], class_id)

      return mask

    except Exception as e:
      logger.warning("Failed to segment {}: {}", frame_path.name, e)
      return None

  def _generate_static_road_mask(self, road_masks: list[np.ndarray]) -> np.ndarray:
    if not road_masks:
      return np.zeros((640, 640), dtype=np.uint8)

    stacked = np.stack(road_masks, axis=0)
    mode_mask = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 0, stacked)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mode_mask = cv2.morphologyEx(mode_mask, cv2.MORPH_CLOSE, kernel)
    mode_mask = cv2.morphologyEx(mode_mask, cv2.MORPH_OPEN, kernel)

    return mode_mask.astype(np.uint8)

  def generate_road_mask(
    self,
    masks: list[np.ndarray],
    class_ids: list[np.ndarray],
  ) -> np.ndarray:
    """Generate static road mask from per-frame masks.

    Args:
        masks: List of per-frame segmentation masks.
        class_ids: List of class ID arrays (not used, kept for API compat).

    Returns:
        Binary road mask.
    """
    road_masks = [m for m in masks if m is not None]
    return self._generate_static_road_mask(road_masks)


def load_annotations(anno_path: str | Path) -> dict:
  """Load segmentation annotations."""
  with open(anno_path) as fh:
    return yaml.safe_load(fh)


def load_road_mask(mask_path: str | Path) -> np.ndarray:
  """Load road mask as binary array."""
  mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
  return (
    (mask > 0).astype(np.uint8)
    if mask is not None
    else np.zeros((640, 640), dtype=np.uint8)
  )


def iter_masks(
  masks_dir: str | Path,
) -> Generator[tuple[np.ndarray, dict], None, None]:
  """Iterate over segmentation masks.

  Args:
      masks_dir: Path to masks directory.

  Yields:
      (mask, info) where mask is HxW uint8 and info contains frame_idx, filename.
  """
  masks_dir = Path(masks_dir)
  mask_files = sorted(masks_dir.glob("*.png"))

  for idx, mask_path in enumerate(mask_files):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is not None:
      yield mask, {"frame_idx": idx, "filename": mask_path.name}
