"""Weights & Biases backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .tracker import BaseLogger


class WandbLogger(BaseLogger):
  """Wrap the ``wandb`` SDK behind the BaseLogger API.

  Pass ``mode="offline"`` to log without network access (useful for CI).
  """

  def __init__(
    self,
    project: str,
    run_name: str | None = None,
    entity: str | None = None,
    mode: str = "online",
    config: dict[str, Any] | None = None,
  ) -> None:
    import wandb

    self.wandb = wandb
    self.run = wandb.init(
      project=project,
      name=run_name,
      entity=entity,
      mode=mode,
      config=config or {},
      reinit=True,
    )

  def log_params(self, params: dict[str, Any]) -> None:
    self.run.config.update(params, allow_val_change=True)

  def log_metric(self, name: str, value: float, step: int | None = None) -> None:
    self.log_metrics({name: float(value)}, step=step)

  def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
    payload = {k: float(v) for k, v in metrics.items()}
    if step is not None:
      self.run.log(payload, step=step)
    else:
      self.run.log(payload)

  def log_artifact(self, path: str | Path, name: str | None = None) -> None:
    p = Path(path)
    artifact_name = name or p.stem
    artifact = self.wandb.Artifact(artifact_name, type=_infer_type(p))
    artifact.add_file(str(p))
    self.run.log_artifact(artifact)

  def close(self) -> None:
    self.run.finish()


def _infer_type(p: Path) -> str:
  suf = p.suffix.lower()
  if suf in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}:
    return "figure"
  if suf in {".csv", ".json", ".jsonl"}:
    return "results"
  if suf in {".pt", ".onnx"}:
    return "model"
  return "file"
