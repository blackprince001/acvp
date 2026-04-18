"""
Temporal window builder (T-058).

Converts a sequence of per-frame feature vectors into (input, target) pairs
for supervised sequence-to-sequence training.
"""

from __future__ import annotations

import numpy as np

from src.utils.types import WindowFeatures


class WindowBuilder:
  """Build sliding windows over a feature sequence.

  Args:
      input_len: Number of frames in the input window (look-back).
      horizon: Number of frames to predict ahead.
      stride: Step size between consecutive windows.
  """

  def __init__(
    self,
    input_len: int = 10,
    horizon: int = 5,
    stride: int = 1,
  ) -> None:
    if input_len < 1 or horizon < 1 or stride < 1:
      raise ValueError("input_len, horizon, and stride must be >= 1")
    self.input_len = input_len
    self.horizon = horizon
    self.stride = stride

  def build(
    self, feature_sequence: list[np.ndarray] | np.ndarray
  ) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) arrays from a full feature sequence.

    Args:
        feature_sequence: List/array of shape (T, F) where T is the number
            of frames and F is the number of features.

    Returns:
        X: shape (N, input_len, F)
        y: shape (N, horizon, F)
    """
    seq = np.asarray(feature_sequence, dtype=np.float64)
    if seq.ndim == 1:
      seq = seq[:, np.newaxis]
    T, F = seq.shape
    min_len = self.input_len + self.horizon
    if T < min_len:
      raise ValueError(
        f"Sequence length {T} is shorter than input_len + horizon = {min_len}"
      )

    X_list, y_list = [], []
    for start in range(0, T - min_len + 1, self.stride):
      X_list.append(seq[start : start + self.input_len])
      y_list.append(seq[start + self.input_len : start + self.input_len + self.horizon])

    return np.stack(X_list), np.stack(y_list)

  def build_single(
    self,
    feature_sequence: list[np.ndarray] | np.ndarray,
    end_idx: int | None = None,
  ) -> WindowFeatures:
    """Build a single input window ending at *end_idx* (for inference).

    Args:
        feature_sequence: Shape (T, F).
        end_idx: Exclusive end index; defaults to len(feature_sequence).

    Returns:
        WindowFeatures with the input window features.
    """
    seq = np.asarray(feature_sequence, dtype=np.float64)
    if seq.ndim == 1:
      seq = seq[:, np.newaxis]
    if end_idx is None:
      end_idx = len(seq)
    start_idx = end_idx - self.input_len
    if start_idx < 0:
      raise ValueError(
        f"Not enough history: need {self.input_len} frames before index {end_idx}"
      )
    window = seq[start_idx:end_idx]
    return WindowFeatures(
      window_start=start_idx,
      window_end=end_idx,
      features=window,
    )
