"""TensorBoard backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .tracker import BaseLogger


class TensorBoardLogger(BaseLogger):
  """Wrap ``torch.utils.tensorboard.SummaryWriter`` behind the BaseLogger API."""

  def __init__(self, log_dir: str | Path) -> None:
    from torch.utils.tensorboard import SummaryWriter

    self.log_dir = Path(log_dir)
    self.log_dir.mkdir(parents=True, exist_ok=True)
    self.writer = SummaryWriter(log_dir=str(self.log_dir))
    self._step_counters: dict[str, int] = {}

  def _step(self, name: str, step: int | None) -> int:
    if step is not None:
      return step
    self._step_counters[name] = self._step_counters.get(name, -1) + 1
    return self._step_counters[name]

  def log_params(self, params: dict[str, Any]) -> None:
    flat = {k: _stringify(v) for k, v in params.items()}
    self.writer.add_hparams(flat, {})

  def log_metric(self, name: str, value: float, step: int | None = None) -> None:
    self.writer.add_scalar(name, float(value), self._step(name, step))

  def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
    for k, v in metrics.items():
      self.log_metric(k, v, step)

  def log_artifact(self, path: str | Path, name: str | None = None) -> None:
    p = Path(path)
    if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
      import numpy as np
      from PIL import Image

      img = np.array(Image.open(p).convert("RGB"))
      tag = name or p.stem
      self.writer.add_image(tag, img, dataformats="HWC")
    # Non-image artifacts are not natively logged by TensorBoard; skip silently.

  def close(self) -> None:
    self.writer.flush()
    self.writer.close()


def _stringify(value: Any) -> Any:
  if isinstance(value, (int, float, bool, str)):
    return value
  return str(value)
