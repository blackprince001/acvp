"""
Evaluation metrics for detection models and ML estimators.

The benchmark runner composes these into per-model result records.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class LatencyStats:
  """Distribution summary for a series of per-iteration latency samples (ms)."""

  mean_ms: float
  std_ms: float
  p50_ms: float
  p90_ms: float
  p95_ms: float
  p99_ms: float
  min_ms: float
  max_ms: float
  n_samples: int
  fps_mean: float

  @classmethod
  def from_samples(cls, samples_ms: list[float] | np.ndarray) -> "LatencyStats":
    arr = np.asarray(samples_ms, dtype=float)
    if arr.size == 0:
      raise ValueError("LatencyStats.from_samples requires at least one sample.")
    return cls(
      mean_ms=float(arr.mean()),
      std_ms=float(arr.std(ddof=0)),
      p50_ms=float(np.percentile(arr, 50)),
      p90_ms=float(np.percentile(arr, 90)),
      p95_ms=float(np.percentile(arr, 95)),
      p99_ms=float(np.percentile(arr, 99)),
      min_ms=float(arr.min()),
      max_ms=float(arr.max()),
      n_samples=int(arr.size),
      fps_mean=float(1000.0 / arr.mean()) if arr.mean() > 0 else 0.0,
    )

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


@dataclass
class DetectionMetrics:
  """Per-model detection scores. Values populated from Ultralytics' validator."""

  precision: float
  recall: float
  map50: float
  map50_95: float
  f1: float
  per_class_ap50: dict[str, float] = field(default_factory=dict)
  per_class_ap50_95: dict[str, float] = field(default_factory=dict)

  @staticmethod
  def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

  @classmethod
  def from_ultralytics(cls, results: Any, class_names: list[str]) -> "DetectionMetrics":
    """Build from an Ultralytics ``DetMetrics`` object (returned by ``model.val()``).

    The Ultralytics object exposes ``.box.mp / .mr / .map50 / .map`` and
    per-class arrays via ``.box.ap50`` and ``.box.ap``.
    """
    box = results.box
    p = float(box.mp)
    r = float(box.mr)
    per_ap50: dict[str, float] = {}
    per_ap: dict[str, float] = {}
    if hasattr(box, "ap50") and hasattr(box, "ap_class_index"):
      for idx, cls_idx in enumerate(box.ap_class_index):
        name = (
          class_names[int(cls_idx)] if int(cls_idx) < len(class_names) else str(cls_idx)
        )
        per_ap50[name] = float(box.ap50[idx])
        per_ap[name] = float(box.ap[idx])
    return cls(
      precision=p,
      recall=r,
      map50=float(box.map50),
      map50_95=float(box.map),
      f1=cls._f1(p, r),
      per_class_ap50=per_ap50,
      per_class_ap50_95=per_ap,
    )

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


@dataclass
class ModelInfo:
  """Static descriptors for a model — cheap to compute once."""

  name: str
  parameters: int
  gflops: float | None
  weights_mb: float

  @classmethod
  def from_ultralytics(
    cls, name: str, model: Any, weights_path: str | Path
  ) -> "ModelInfo":
    n_params = sum(p.numel() for p in model.parameters())
    weights_mb = Path(weights_path).stat().st_size / (1024 * 1024)
    gflops: float | None = None
    try:
      info = model.info(detailed=False, verbose=False)
      if isinstance(info, tuple) and len(info) >= 4:
        gflops = float(info[3])
    except Exception:  # noqa: BLE001
      gflops = None
    return cls(
      name=name, parameters=int(n_params), gflops=gflops, weights_mb=weights_mb
    )

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


@dataclass
class RegressionMetrics:
  """Standard regression scores for forecast vs. ground-truth telemetry."""

  mae: float
  rmse: float
  mape: float
  r2: float
  n: int

  @classmethod
  def from_arrays(cls, y_true: np.ndarray, y_pred: np.ndarray) -> "RegressionMetrics":
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
      raise ValueError(
        f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
      )
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, y_true)
    mape = float(np.nanmean(np.abs(err / denom)) * 100.0)
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return cls(mae=mae, rmse=rmse, mape=mape, r2=r2, n=int(y_true.size))

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)
