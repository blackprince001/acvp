"""
Base detection model interface.

All detection model wrappers (YOLOv8, YOLO11, etc.) must subclass this
and implement the three core methods: train(), predict(), and export_onnx().
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BaseDetectionModel(ABC):
    """Abstract base class defining the detection model interface.

    Args:
        config: Model configuration dict (typically loaded from a YAML config).
        device: Compute device string, e.g. ``"cuda"`` or ``"cpu"``.
    """

    def __init__(self, config: dict, device: str = "cuda") -> None:
        self.config = config
        self.device = device
        self.model = None  # Underlying framework model, set in subclass

    @abstractmethod
    def train(
        self,
        data_yaml: str | Path,
        output_dir: str | Path,
        **kwargs: Any,
    ) -> dict:
        """Fine-tune the model on the provided dataset.

        Args:
            data_yaml: Path to a YOLO-format ``data.yaml`` describing the
                dataset (train/val/test split paths and class names).
            output_dir: Directory where checkpoints and logs are saved.
            **kwargs: Additional training overrides forwarded to the framework
                trainer (e.g. ``epochs``, ``batch``).

        Returns:
            A ``dict`` containing at minimum ``{"best_checkpoint": Path, "metrics": dict}``.
        """
        ...

    @abstractmethod
    def predict(
        self,
        source: str | Path | np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45,
        **kwargs: Any,
    ) -> list[dict]:
        """Run inference on *source*.

        Args:
            source: A file path, directory, URL, or raw NumPy image array.
            conf: Confidence threshold for detections.
            iou: NMS IoU threshold.
            **kwargs: Additional inference parameters.

        Returns:
            A list of per-image result dicts with keys:
            ``{"boxes": np.ndarray, "scores": np.ndarray, "class_ids": np.ndarray}``.
        """
        ...

    @abstractmethod
    def export_onnx(
        self,
        output_path: str | Path,
        image_size: int = 640,
        dynamic: bool = True,
        **kwargs: Any,
    ) -> Path:
        """Export the model to ONNX format.

        Args:
            output_path: Destination ``.onnx`` file path.
            image_size: Input resolution used for export.
            dynamic: Whether to use dynamic batch/resolution axes.
            **kwargs: Additional export parameters.

        Returns:
            Resolved path of the exported ``.onnx`` file.
        """
        ...

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load model weights from a saved checkpoint.

        Args:
            checkpoint_path: Path to a ``.pt`` checkpoint file.
        """
        ...

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _training_params(self, overrides: dict | None = None) -> dict:
        """Merge config training params with any caller-supplied overrides.

        Args:
            overrides: Optional dict of override values.

        Returns:
            Merged training parameter dict.
        """
        params = dict(self.config.get("training", {}))
        if overrides:
            params.update(overrides)
        return params

    def __repr__(self) -> str:
        # Config may have "name" at top level (flattened) or nested under "model"
        name = (
            self.config.get("name")
            or self.config.get("model", {}).get("name")
            or self.__class__.__name__
        )
        return f"{self.__class__.__name__}(model={name!r}, device={self.device!r})"
