"""
Road mask manager.

Loads a static road mask from disk, generates one from segmentation output,
and returns the appropriate mask per frame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class RoadMaskManager:
  """Manages road masks for spatial filtering.

  A static mask is a single binary array applied to every frame.
  Per-frame masks (from segmentation) override the static mask for that frame.

  Args:
      default_shape: (H, W) used when no mask has been loaded yet.
  """

  def __init__(self, default_shape: tuple[int, int] | None = None) -> None:
    self._static_mask: np.ndarray | None = None
    self._per_frame: dict[int, np.ndarray] = {}
    self._default_shape = default_shape

  # ------------------------------------------------------------------
  # Loading / generation
  # ------------------------------------------------------------------

  def load_static(self, path: str | Path) -> np.ndarray:
    """Load a static road mask from a file.

    Supports .npy (binary array) and image files (any format readable
    by PIL/OpenCV).  The mask is binarised: non-zero pixels → True.

    Args:
        path: Path to the mask file.

    Returns:
        Boolean ndarray of shape (H, W).
    """
    path = Path(path)
    if path.suffix == ".npy":
      mask = np.load(path)
    else:
      from PIL import Image  # lazy import – not always needed

      mask = np.array(Image.open(path).convert("L"))

    self._static_mask = mask.astype(bool)
    return self._static_mask

  def generate_from_segmentation(
    self,
    segmentation_masks: list[np.ndarray],
    frame_id: int | None = None,
  ) -> np.ndarray:
    """Generate a road mask by merging segmentation masks.

    Performs a logical OR across all provided masks.  If *frame_id* is
    given the result is stored as a per-frame mask; otherwise it replaces
    the static mask.

    Args:
        segmentation_masks: List of boolean/uint8 arrays, each (H, W).
        frame_id: Optional frame index to store the mask against.

    Returns:
        Boolean ndarray of shape (H, W).
    """
    if not segmentation_masks:
      raise ValueError("segmentation_masks must not be empty")

    merged = np.zeros_like(segmentation_masks[0], dtype=bool)
    for m in segmentation_masks:
      merged |= m.astype(bool)

    if frame_id is not None:
      self._per_frame[frame_id] = merged
    else:
      self._static_mask = merged

    return merged

  # ------------------------------------------------------------------
  # Retrieval
  # ------------------------------------------------------------------

  def get_mask(self, frame_id: int | None = None) -> np.ndarray:
    """Return the road mask for a given frame.

    Per-frame masks take priority over the static mask.  If neither
    exists and *default_shape* was provided, an all-True mask is returned.

    Args:
        frame_id: Frame index.  Pass None to get the static mask.

    Returns:
        Boolean ndarray of shape (H, W).

    Raises:
        RuntimeError: If no mask is available and no default shape set.
    """
    if frame_id is not None and frame_id in self._per_frame:
      return self._per_frame[frame_id]
    if self._static_mask is not None:
      return self._static_mask
    if self._default_shape is not None:
      return np.ones(self._default_shape, dtype=bool)
    raise RuntimeError(
      "No road mask available. Call load_static() or generate_from_segmentation() first."
    )

  def set_static(self, mask: np.ndarray) -> None:
    """Directly set the static mask (useful in tests / pipelines).

    Args:
        mask: Boolean or uint8 array of shape (H, W).
    """
    self._static_mask = mask.astype(bool)

  def set_frame_mask(self, frame_id: int, mask: np.ndarray) -> None:
    """Store a per-frame mask.

    Args:
        frame_id: Frame index.
        mask: Boolean or uint8 array of shape (H, W).
    """
    self._per_frame[frame_id] = mask.astype(bool)
