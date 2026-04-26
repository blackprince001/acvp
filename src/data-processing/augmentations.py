"""Training augmentations for traffic data."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torchvision.transforms.functional as TF
from loguru import logger


@dataclass
class AugmentationConfig:
  """Configuration for data augmentations."""

  brightness: float = 0.0
  contrast: float = 0.0
  blur: float = 0.0
  rotation: float = 0.0
  flip_prob: float = 0.0


def get_train_transforms(config: AugmentationConfig | None = None) -> Callable:
  """Get augmentation pipeline for training.

  Args:
      config: Augmentation parameters.

  Returns:
      Transform function.
  """
  return TrafficAugmentation(config or AugmentationConfig())


class TrafficAugmentation:
  """Training augmentations for traffic images.

  Implements:
  - Random brightness/contrast (lighting variation)
  - Random Gaussian blur (camera quality variation)
  - Random rotation (camera angle variation)
  - Random horizontal flip (symmetric road views)
  """

  def __init__(self, config: AugmentationConfig) -> None:
    self.brightness = config.brightness
    self.contrast = config.contrast
    self.blur = config.blur
    self.rotation = config.rotation
    self.flip_prob = config.flip_prob

  def __call__(
    self,
    image: torch.Tensor,
    boxes: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    masks: torch.Tensor | None = None,
  ):
    """Apply augmentations to image and optionally boxes/labels/masks.

    Args:
        image: Tensor of shape (C, H, W).
        boxes: Tensor of shape (N, 4) in xyxy format.
        labels: Tensor of shape (N,).
        masks: Tensor of shape (H, W) or (N, H, W).

    Returns:
        Augmented tensors. If masks provided, returns (image, boxes, labels, masks).
        Otherwise returns (image, boxes, labels).
    """
    if not isinstance(image, torch.Tensor):
      raise TypeError("Image must be a torch.Tensor")

    image = image.clone()

    if random.random() < self.blur:
      image = self._apply_blur(image)

    if self.brightness > 0:
      factor = 1.0 + random.uniform(-self.brightness, self.brightness)
      image = TF.adjust_brightness(image, factor)

    if self.contrast > 0:
      factor = 1.0 + random.uniform(-self.contrast, self.contrast)
      image = TF.adjust_contrast(image, factor)

    if self.rotation > 0:
      angle = random.uniform(-self.rotation, self.rotation)
      image, boxes = self._apply_rotation(image, boxes, angle)

    if random.random() < self.flip_prob:
      image, boxes = self._apply_horizontal_flip(image, boxes)
      if masks is not None:
        masks = torch.flip(masks, dims=[-1])

    if masks is not None:
      return image, boxes, labels, masks
    return image, boxes, labels

  def _apply_blur(self, image: torch.Tensor) -> torch.Tensor:
    if random.random() < 0.5:
      kernel_size = random.choice([3, 5])
      image = TF.gaussian_blur(image, kernel_size=kernel_size)
    return image

  def _apply_rotation(
    self,
    image: torch.Tensor,
    boxes: torch.Tensor | None,
    angle: float,
  ) -> tuple[torch.Tensor, torch.Tensor | None]:
    _, h, w = image.shape
    image = TF.rotate(image, angle)

    if boxes is not None and len(boxes) > 0:
      cx = (boxes[:, 0] + boxes[:, 2]) / 2 / w
      cy = (boxes[:, 1] + boxes[:, 3]) / 2 / h

      angle_rad = np.radians(angle)
      cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

      new_cx = cx * cos_a - cy * sin_a
      new_cy = cx * sin_a + cy * cos_a

      new_cx = np.clip(new_cx, 0, 1)
      new_cy = np.clip(new_cy, 0, 1)

      cx = new_cx * w
      cy = new_cy * h
      half_w = (boxes[:, 2] - boxes[:, 0]) / 2
      half_h = (boxes[:, 3] - boxes[:, 1]) / 2

      boxes = torch.stack(
        [
          cx - half_w,
          cy - half_h,
          cx + half_w,
          cy + half_h,
        ],
        dim=1,
      )

      boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
      boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)

    return image, boxes

  def _apply_horizontal_flip(
    self,
    image: torch.Tensor,
    boxes: torch.Tensor | None,
  ) -> tuple[torch.Tensor, torch.Tensor | None]:
    image = TF.hflip(image)

    if boxes is not None and len(boxes) > 0:
      _, _, w = image.shape
      boxes = torch.stack(
        [
          w - boxes[:, 2],
          boxes[:, 1],
          w - boxes[:, 0],
          boxes[:, 3],
        ],
        dim=1,
      )

    return image, boxes


class ComposeTransforms:
  """Compose multiple transform functions."""

  def __init__(self, transforms: list[Callable]) -> None:
    self.transforms = transforms

  def __call__(self, *args, **kwargs):
    result = args
    for t in self.transforms:
      result = t(*result, **kwargs)
    return result


def get_validation_transforms() -> Callable:
  """Get validation transforms (identity, no augmentation)."""

  def noop(image, boxes=None, labels=None, masks=None):
    if masks is not None:
      return image, boxes, labels, masks
    return image, boxes, labels

  return noop
