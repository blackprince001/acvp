"""PyTorch Dataset implementations for traffic data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import Dataset


@dataclass
class DetectionSample:
  """Single detection dataset sample."""

  image: torch.Tensor
  boxes: torch.Tensor
  labels: torch.Tensor
  image_id: str


@dataclass
class SegmentationSample:
  """Single segmentation dataset sample."""

  image: torch.Tensor
  boxes: torch.Tensor
  labels: torch.Tensor
  masks: torch.Tensor
  image_id: str


@dataclass
class TelemetrySample:
  """Single telemetry dataset sample."""

  input_window: torch.Tensor
  target_window: torch.Tensor


class TrafficDetectionDataset(Dataset):
  """Dataset for YOLO detection training.

  Expected structure:
      frames_dir/
          frame_000001.jpg
          frame_000002.jpg
          ...
      annotations.json:
          [
              {"image_id": "frame_000001", "boxes": [[x1,y1,x2,y2], ...], "labels": [0,1,...]},
              ...
          ]
  """

  def __init__(
    self,
    frames_dir: str | Path,
    annotations_path: str | Path | None = None,
    transforms: Callable | None = None,
    image_size: tuple[int, int] = (640, 640),
  ) -> None:
    self.frames_dir = Path(frames_dir)
    self.transforms = transforms
    self.image_size = image_size

    if annotations_path is None:
      annotations_path = self.frames_dir.parent / "annotations.json"
    self.annotations_path = Path(annotations_path)

    self.samples = self._load_annotations()

    logger.info(
      "TrafficDetectionDataset: {} samples from {}",
      len(self.samples),
      self.frames_dir,
    )

  def _load_annotations(self) -> list[dict]:
    if not self.annotations_path.exists():
      return []

    with open(self.annotations_path) as fh:
      return json.load(fh)

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(self, idx: int) -> DetectionSample:
    sample = self.samples[idx]
    image_id = sample["image_id"]

    img_path = self.frames_dir / f"{image_id}.jpg"
    if not img_path.exists():
      img_path = self.frames_dir / image_id

    image = cv2.imread(str(img_path))
    if image is None:
      raise FileNotFoundError(f"Cannot load image: {img_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, self.image_size)
    image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

    boxes = torch.tensor(sample.get("boxes", []), dtype=torch.float32)
    labels = torch.tensor(sample.get("labels", []), dtype=torch.int64)

    if self.transforms is not None:
      image, boxes, labels = self.transforms(image, boxes, labels)

    return DetectionSample(image=image, boxes=boxes, labels=labels, image_id=image_id)


class TrafficSegmentationDataset(Dataset):
  """Dataset for YOLO segmentation training.

  Expected structure:
      frames_dir/
          frame_000001.jpg
          ...
      masks_dir/
          frame_000001.png
          ...
      annotations.json (optional)
  """

  def __init__(
    self,
    frames_dir: str | Path,
    masks_dir: str | Path,
    annotations_path: str | Path | None = None,
    transforms: Callable | None = None,
    image_size: tuple[int, int] = (640, 640),
  ) -> None:
    self.frames_dir = Path(frames_dir)
    self.masks_dir = Path(masks_dir)
    self.transforms = transforms
    self.image_size = image_size

    if annotations_path is None:
      annotations_path = self.frames_dir.parent / "annotations.json"
    self.annotations_path = Path(annotations_path)

    self.samples = self._load_annotations()

    logger.info(
      "TrafficSegmentationDataset: {} samples from {}",
      len(self.samples),
      self.frames_dir,
    )

  def _load_annotations(self) -> list[dict]:
    if not self.annotations_path.exists():
      frame_files = sorted(self.frames_dir.glob("*.jpg"))
      return [{"image_id": f.stem} for f in frame_files]

    with open(self.annotations_path) as fh:
      return json.load(fh)

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(self, idx: int) -> SegmentationSample:
    sample = self.samples[idx]
    image_id = sample["image_id"]

    img_path = self.frames_dir / f"{image_id}.jpg"
    if not img_path.exists():
      img_path = self.frames_dir / image_id

    image = cv2.imread(str(img_path))
    if image is None:
      raise FileNotFoundError(f"Cannot load image: {img_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, self.image_size)
    image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

    mask_path = self.masks_dir / f"{image_id}.png"
    if mask_path.exists():
      mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
      mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
      masks = torch.from_numpy(mask).long()
    else:
      masks = torch.zeros(self.image_size, dtype=torch.long)

    boxes = torch.tensor(sample.get("boxes", []), dtype=torch.float32)
    labels = torch.tensor(sample.get("labels", []), dtype=torch.int64)

    if self.transforms is not None:
      image, boxes, labels, masks = self.transforms(image, boxes, labels, masks)

    return SegmentationSample(
      image=image, boxes=boxes, labels=labels, masks=masks, image_id=image_id
    )


class TelemetryDataset(Dataset):
  """Dataset for ML Estimator training from telemetry CSV/Parquet.

  Expected telemetry columns:
      timestamp, vehicle_count, occupancy_ratio, avg_velocity,
      density_index, flow_rate, congestion_level, ...
  """

  def __init__(
    self,
    telemetry_path: str | Path,
    window_size: int = 30,
    prediction_horizon: int = 5,
    stride: int = 1,
    target_columns: list[str] | None = None,
  ) -> None:
    self.telemetry_path = Path(telemetry_path)
    self.window_size = window_size
    self.prediction_horizon = prediction_horizon
    self.stride = stride

    self.data = self._load_telemetry()
    self.feature_columns = [c for c in self.data.columns if c != "timestamp"]

    if target_columns is None:
      target_columns = ["vehicle_count", "occupancy_ratio"]
    self.target_columns = target_columns

    self.input_size = len(self.feature_columns)
    self.output_size = len(self.target_columns) * prediction_horizon

    self.valid_indices = self._compute_valid_indices()

    logger.info(
      "TelemetryDataset: {} windows from {} rows, input={}, output={}",
      len(self.valid_indices),
      len(self.data),
      self.input_size,
      self.output_size,
    )

  def _load_telemetry(self) -> pd.DataFrame:
    if self.telemetry_path.suffix == ".parquet":
      import pandas as pd

      return pd.read_parquet(self.telemetry_path)
    else:
      import pandas as pd

      return pd.read_csv(self.telemetry_path)

  def _compute_valid_indices(self) -> list[int]:
    total = len(self.data)
    valid = []
    i = 0
    while i + self.window_size + self.prediction_horizon <= total:
      valid.append(i)
      i += self.stride
    return valid

  def __len__(self) -> int:
    return len(self.valid_indices)

  def __getitem__(self, idx: int) -> TelemetrySample:
    start_idx = self.valid_indices[idx]
    input_end = start_idx + self.window_size
    target_end = input_end + self.prediction_horizon

    input_data = self.data.iloc[start_idx:input_end][self.feature_columns].values
    target_data = self.data.iloc[input_end:target_end][self.target_columns].values

    input_window = torch.from_numpy(input_data).float()
    target_window = torch.from_numpy(target_data).float()

    return TelemetrySample(input_window=input_window, target_window=target_window)


def detection_collate_fn(batch: list[DetectionSample]) -> dict:
  """Collate function for detection samples."""
  images = torch.stack([s.image for s in batch])
  image_ids = [s.image_id for s in batch]

  boxes = [s.boxes for s in batch]
  labels = [s.labels for s in batch]

  return {"images": images, "boxes": boxes, "labels": labels, "image_ids": image_ids}


def segmentation_collate_fn(batch: list[SegmentationSample]) -> dict:
  """Collate function for segmentation samples."""
  images = torch.stack([s.image for s in batch])
  image_ids = [s.image_id for s in batch]

  boxes = [s.boxes for s in batch]
  labels = [s.labels for s in batch]
  masks = torch.stack([s.masks for s in batch])

  return {
    "images": images,
    "boxes": boxes,
    "labels": labels,
    "masks": masks,
    "image_ids": image_ids,
  }


def telemetry_collate_fn(batch: list[TelemetrySample]) -> TelemetrySample:
  """Collate function for telemetry samples."""
  input_windows = torch.stack([s.input_window for s in batch])
  target_windows = torch.stack([s.target_window for s in batch])
  return TelemetrySample(input_window=input_windows, target_window=target_windows)
