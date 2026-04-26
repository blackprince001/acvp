"""
Base ML estimator interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class BaseEstimator(nn.Module, ABC):
  """Abstract base for sequence-to-sequence traffic density estimators.

  Args:
      config: Dict with at minimum:
          - input_size (int): number of input features per timestep
          - output_size (int): number of output features per timestep
          - horizon (int): number of future timesteps to predict
  """

  def __init__(self, config: dict[str, Any]) -> None:
    super().__init__()
    self.config = config
    self.input_size: int = config["input_size"]
    self.output_size: int = config["output_size"]
    self.horizon: int = config["horizon"]

  @abstractmethod
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass.

    Args:
        x: Input tensor of shape (B, T, input_size).

    Returns:
        Output tensor of shape (B, horizon * output_size).
    """
    ...

  def predict(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
    """Run inference without gradient tracking.

    Args:
        x: Array/tensor of shape (B, T, input_size) or (T, input_size).

    Returns:
        Numpy array of shape (B, horizon, output_size).
    """
    self.eval()
    if not isinstance(x, torch.Tensor):
      x = torch.tensor(x, dtype=torch.float32)
    if x.ndim == 2:
      x = x.unsqueeze(0)
    with torch.no_grad():
      out = self.forward(x)  # (B, horizon * output_size)
    return out.reshape(-1, self.horizon, self.output_size).cpu().numpy()
