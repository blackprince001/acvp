"""
Detection model registry.

Config-based factory that returns the correct model instance by inspecting the
``model.name`` field of the configuration dict.

Supported model families (all share the same Ultralytics YOLO API):

+---------------+----------------------------+----------------------------------+
| Family        | Weight name examples       | Class                            |
+===============+============================+==================================+
| YOLOv5        | yolov5n/s/m/l/x            | YOLOv5DetectionModel             |
+---------------+----------------------------+----------------------------------+
| YOLOv8        | yolov8n/s/m/l/x            | YOLOv8DetectionModel             |
+---------------+----------------------------+----------------------------------+
| YOLOv9        | yolov9s/m/c/e              | YOLOv9DetectionModel             |
+---------------+----------------------------+----------------------------------+
| YOLOv10       | yolov10n/s/m/l/x/b         | YOLOv10DetectionModel            |
+---------------+----------------------------+----------------------------------+
| YOLO11        | yolo11n/s/m/l/x            | YOLO11DetectionModel             |
+---------------+----------------------------+----------------------------------+
| RT-DETR       | rtdetr-l/x                 | RTDETRDetectionModel             |
+---------------+----------------------------+----------------------------------+
| Generic       | any other weight string    | YOLODetectionModel               |
+---------------+----------------------------+----------------------------------+

Usage::

    from src.models.detection.registry import get_model

    model = get_model("yolov9s", device="cuda")
    model = get_model("yolov10n", device="cuda")
    model = get_model("rtdetr-l", device="cuda")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .base import BaseDetectionModel
from .rtdetr import RTDETRDetectionModel
from .yolo import YOLODetectionModel
from .yolov5 import YOLOv5DetectionModel
from .yolov8 import YOLOv8DetectionModel
from .yolov9 import YOLOv9DetectionModel
from .yolov10 import YOLOv10DetectionModel
from .yolov11 import YOLO11DetectionModel

_REGISTRY: dict[str, type[BaseDetectionModel]] = {
  "yolov5": YOLOv5DetectionModel,
  "yolov8": YOLOv8DetectionModel,
  "yolov9": YOLOv9DetectionModel,
  "yolov10": YOLOv10DetectionModel,
  "yolov11": YOLO11DetectionModel,
  "yolo11": YOLO11DetectionModel,  # Ultralytics native alias
  "rtdetr": RTDETRDetectionModel,
  "yolo": YOLODetectionModel,  # generic fallback
}

# Default config directory (relative to project root)
_DEFAULT_CONFIG_DIR = Path("configs/models/detection")


def get_model(
  model_name: str,
  config: dict | None = None,
  config_path: str | Path | None = None,
  device: str = "cuda",
  checkpoint_path: str | Path | None = None,
  **kwargs: Any,
) -> BaseDetectionModel:
  """Instantiate a detection model by name.

  Looks up *model_name* in the internal registry (longest-prefix match),
  loads the config from *config_path* (or auto-discovers it under the
  default config dir) if *config* is not supplied, then constructs and
  returns the model.

  Args:
      model_name: Model identifier — e.g. ``"yolov8n"``, ``"yolo11s"``,
          ``"yolov9c"``, ``"yolov10m"``, ``"rtdetr-l"``.
      config: Pre-loaded configuration dict. If supplied, *config_path*
          is ignored.
      config_path: Explicit path to a YAML config file. Falls back to
          ``configs/models/detection/<model_name>.yaml`` if not given.
      device: Compute device (``"cuda"``, ``"cpu"``, ``"mps"``).
      checkpoint_path: Optional path to a ``.pt`` checkpoint.
      **kwargs: Extra kwargs forwarded to the model constructor.

  Returns:
      An instantiated :class:`BaseDetectionModel` subclass.

  Raises:
      ValueError: If *model_name* does not match any registered prefix.
      FileNotFoundError: If an auto-discovered config YAML does not exist.

  Example::

      model = get_model("yolov9s", device="cuda")
      model = get_model("rtdetr-l", device="cuda")
      # With an inline config (no YAML file needed):
      model = get_model("yolov10n", config={"name": "yolov10n"})
  """
  model_name_lower = model_name.lower()

  # Longest-prefix match to avoid e.g. "yolov10" matching "yolov1"
  model_cls: type[BaseDetectionModel] | None = None
  for prefix in sorted(_REGISTRY, key=len, reverse=True):
    if model_name_lower.startswith(prefix):
      model_cls = _REGISTRY[prefix]
      break
  if model_cls is None:
    registered = sorted(_REGISTRY.keys())
    raise ValueError(
      f"Unknown model name {model_name!r}. Registered prefixes: {registered}"
    )

  # Resolve config
  if config is None:
    if config_path is None:
      config_path = _DEFAULT_CONFIG_DIR / f"{model_name_lower}.yaml"
    config_path = Path(config_path)
    if not config_path.exists():
      raise FileNotFoundError(
        f"Config file not found for model {model_name!r}: {config_path}"
      )
    with open(config_path) as f:
      raw = yaml.safe_load(f)
    # Flatten model + training into a single dict
    config = {**(raw.get("model", {})), "training": raw.get("training", {})}

  return model_cls(
    config=config, device=device, checkpoint_path=checkpoint_path, **kwargs
  )


def list_models() -> list[str]:
  """Return sorted list of registered model name prefixes."""
  return sorted(k for k in _REGISTRY if k != "yolo")  # hide the raw generic
