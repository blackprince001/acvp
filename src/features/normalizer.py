"""
Feature normalizer (T-059).

Supports standard (z-score), minmax, and robust normalization.
Pure numpy — no sklearn dependency.
"""

from __future__ import annotations

import numpy as np


class FeatureNormalizer:
  """Fit-transform normalizer for feature matrices.

  Args:
      method: One of "standard", "minmax", "robust".
  """

  _METHODS = {"standard", "minmax", "robust"}

  def __init__(self, method: str = "standard") -> None:
    if method not in self._METHODS:
      raise ValueError(f"method must be one of {self._METHODS}, got {method!r}")
    self.method = method
    self._fitted = False
    self._center: np.ndarray | None = None
    self._scale: np.ndarray | None = None

  def fit(self, X: np.ndarray) -> "FeatureNormalizer":
    """Compute normalization statistics from X.

    Args:
        X: Array of shape (N, F).

    Returns:
        self
    """
    X = np.asarray(X, dtype=np.float64)
    if self.method == "standard":
      self._center = X.mean(axis=0)
      self._scale = X.std(axis=0)
    elif self.method == "minmax":
      self._center = X.min(axis=0)
      self._scale = X.max(axis=0) - X.min(axis=0)
    else:  # robust
      self._center = np.median(X, axis=0)
      q75 = np.percentile(X, 75, axis=0)
      q25 = np.percentile(X, 25, axis=0)
      self._scale = q75 - q25

    # Avoid division by zero
    self._scale = np.where(self._scale == 0, 1.0, self._scale)
    self._fitted = True
    return self

  def transform(self, X: np.ndarray) -> np.ndarray:
    """Normalize X using fitted statistics.

    Args:
        X: Array of shape (N, F) or (F,).

    Returns:
        Normalized array, same shape as X.
    """
    self._check_fitted()
    return (np.asarray(X, dtype=np.float64) - self._center) / self._scale

  def inverse_transform(self, X: np.ndarray) -> np.ndarray:
    """Reverse normalization.

    Args:
        X: Normalized array of shape (N, F) or (F,).

    Returns:
        Original-scale array, same shape as X.
    """
    self._check_fitted()
    return np.asarray(X, dtype=np.float64) * self._scale + self._center

  def fit_transform(self, X: np.ndarray) -> np.ndarray:
    """Fit then transform X.

    Args:
        X: Array of shape (N, F).

    Returns:
        Normalized array of shape (N, F).
    """
    return self.fit(X).transform(X)

  def _check_fitted(self) -> None:
    if not self._fitted:
      raise RuntimeError("Call fit() before transform().")
