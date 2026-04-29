"""
Unified experiment tracker.

A thin facade over pluggable backends. All backends implement the same
interface so callers don't depend on a specific one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseLogger(ABC):
  """Interface every backend implements."""

  @abstractmethod
  def log_params(self, params: dict[str, Any]) -> None: ...

  @abstractmethod
  def log_metric(self, name: str, value: float, step: int | None = None) -> None: ...

  @abstractmethod
  def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...

  @abstractmethod
  def log_artifact(self, path: str | Path, name: str | None = None) -> None: ...

  @abstractmethod
  def close(self) -> None: ...


class NoOpLogger(BaseLogger):
  """Logger that swallows everything. Used when tracking is disabled."""

  def log_params(self, params: dict[str, Any]) -> None:
    return

  def log_metric(self, name: str, value: float, step: int | None = None) -> None:
    return

  def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
    return

  def log_artifact(self, path: str | Path, name: str | None = None) -> None:
    return

  def close(self) -> None:
    return


class ExperimentTracker(BaseLogger):
  """Fan-out logger: forwards every call to all configured backends.

  Construct with a list of backend instances. Failures in one backend
  never stop the others.
  """

  def __init__(self, backends: list[BaseLogger] | None = None) -> None:
    self.backends: list[BaseLogger] = backends or [NoOpLogger()]

  @classmethod
  def from_config(cls, config: dict[str, Any]) -> "ExperimentTracker":
    """Build a tracker from a config dict.

    No backends are currently registered, so this always returns a no-op
    tracker. The method is kept for forward-compatibility.
    """
    return cls([NoOpLogger()])

  def _safe(self, name: str, *args, **kwargs) -> None:
    for backend in self.backends:
      try:
        getattr(backend, name)(*args, **kwargs)
      except Exception as exc:  # noqa: BLE001
        from loguru import logger

        logger.warning(
          "Tracker backend {} failed on {}: {}", type(backend).__name__, name, exc
        )

  def log_params(self, params: dict[str, Any]) -> None:
    self._safe("log_params", params)

  def log_metric(self, name: str, value: float, step: int | None = None) -> None:
    self._safe("log_metric", name, value, step)

  def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
    self._safe("log_metrics", metrics, step)

  def log_artifact(self, path: str | Path, name: str | None = None) -> None:
    self._safe("log_artifact", path, name)

  def close(self) -> None:
    self._safe("close")
