"""Unit tests for ML estimator models."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ml_estimator.gru import GRUEstimator
from src.models.ml_estimator.lstm import LSTMEstimator
from src.models.ml_estimator.tcn import TCNEstimator


def _cfg(**overrides) -> dict:
  base = {
    "input_size": 10,
    "output_size": 10,
    "horizon": 5,
    "hidden_size": 32,
    "num_layers": 2,
    "dropout": 0.0,
    "bidirectional": False,
    "num_channels": [32, 32],
    "kernel_size": 3,
  }
  base.update(overrides)
  return base


def _x(B: int = 4, T: int = 10, F: int = 10) -> torch.Tensor:
  return torch.randn(B, T, F)


class TestLSTM:
  def test_output_shape(self):
    m = LSTMEstimator(_cfg())
    out = m(_x())
    assert out.shape == (4, 5 * 10)

  def test_bidirectional_output_shape(self):
    m = LSTMEstimator(_cfg(bidirectional=True))
    out = m(_x())
    assert out.shape == (4, 5 * 10)

  def test_single_layer_no_dropout(self):
    m = LSTMEstimator(_cfg(num_layers=1, dropout=0.5))
    out = m(_x())
    assert out.shape == (4, 5 * 10)

  def test_gradient_flow(self):
    m = LSTMEstimator(_cfg())
    out = m(_x())
    loss = out.sum()
    loss.backward()
    for p in m.parameters():
      assert p.grad is not None

  def test_predict_returns_numpy(self):
    m = LSTMEstimator(_cfg())
    result = m.predict(_x().numpy())
    assert isinstance(result, np.ndarray)
    assert result.shape == (4, 5, 10)

  def test_predict_2d_input(self):
    m = LSTMEstimator(_cfg())
    result = m.predict(torch.randn(10, 10))
    assert result.shape == (1, 5, 10)


class TestGRU:
  def test_output_shape(self):
    m = GRUEstimator(_cfg())
    out = m(_x())
    assert out.shape == (4, 5 * 10)

  def test_bidirectional_output_shape(self):
    m = GRUEstimator(_cfg(bidirectional=True))
    out = m(_x())
    assert out.shape == (4, 5 * 10)

  def test_single_layer(self):
    m = GRUEstimator(_cfg(num_layers=1))
    out = m(_x())
    assert out.shape == (4, 5 * 10)

  def test_gradient_flow(self):
    m = GRUEstimator(_cfg())
    out = m(_x())
    out.sum().backward()
    for p in m.parameters():
      assert p.grad is not None

  def test_predict_returns_numpy(self):
    m = GRUEstimator(_cfg())
    result = m.predict(_x().numpy())
    assert isinstance(result, np.ndarray)
    assert result.shape == (4, 5, 10)


class TestTCN:
  def test_output_shape(self):
    m = TCNEstimator(_cfg())
    out = m(_x())
    assert out.shape == (4, 5 * 10)

  def test_single_block(self):
    m = TCNEstimator(_cfg(num_channels=[32]))
    out = m(_x())
    assert out.shape == (4, 5 * 10)

  def test_deep_network(self):
    m = TCNEstimator(_cfg(num_channels=[16, 32, 64, 64]))
    out = m(_x())
    assert out.shape == (4, 5 * 10)

  def test_gradient_flow(self):
    m = TCNEstimator(_cfg())
    out = m(_x())
    out.sum().backward()
    for p in m.parameters():
      assert p.grad is not None

  def test_predict_returns_numpy(self):
    m = TCNEstimator(_cfg())
    result = m.predict(_x().numpy())
    assert isinstance(result, np.ndarray)
    assert result.shape == (4, 5, 10)

  def test_different_horizon(self):
    m = TCNEstimator(_cfg(horizon=3, output_size=5))
    out = m(_x(F=10))
    assert out.shape == (4, 3 * 5)

  def test_predict_2d_input(self):
    m = TCNEstimator(_cfg())
    result = m.predict(torch.randn(10, 10))
    assert result.shape == (1, 5, 10)
