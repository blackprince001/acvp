"""
Generic Ultralytics YOLO detection wrapper.

All current Ultralytics YOLO variants (YOLOv5, YOLOv8, YOLOv9, YOLOv10,
YOLO11, YOLO-World, RT-DETR, etc.) share the same ``YOLO()`` class and API.
This module provides a single, unified wrapper that works for all of them —
no matter which weight string you pass.

Model-family-specific subclasses (``YOLOv8DetectionModel``,
``YOLO11DetectionModel``, etc.) simply inherit from this class to keep the
public API consistent with whatever was already imported elsewhere.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseDetectionModel


class YOLODetectionModel(BaseDetectionModel):
    """Generic wrapper for any Ultralytics YOLO detection model.

    Supports all weight families available through Ultralytics, including:
    - **YOLOv5** — ``yolov5n/s/m/l/x.pt``
    - **YOLOv8** — ``yolov8n/s/m/l/x.pt``
    - **YOLOv9** — ``yolov9s/m/c/e.pt``
    - **YOLOv10** — ``yolov10n/s/m/l/x/b.pt``
    - **YOLO11** — ``yolo11n/s/m/l/x.pt``
    - **RT-DETR** — ``rtdetr-l/x.pt``
    - **YOLO-World** — ``yolov8s-world.pt``

    Args:
        config: Model configuration dict with at minimum a ``name`` key and
            (optionally) a ``training`` sub-dict.
        device: Compute device (``"cuda"``, ``"cpu"``, ``"mps"``).
        checkpoint_path: Optional path to a ``.pt`` checkpoint to resume from
            instead of loading the base pre-trained weights.

    Example::

        model = YOLODetectionModel({"name": "yolov9s", "pretrained_weights": "yolov9s.pt"}, device="cuda")
        model.train("data/yolo_detection/data.yaml", "experiments/detection/yolov9s")
    """

    def __init__(
        self,
        config: dict,
        device: str = "cuda",
        checkpoint_path: str | Path | None = None,
    ) -> None:
        super().__init__(config, device)
        from ultralytics import YOLO  # lazy import

        model_cfg = config.get("model", config)
        name = model_cfg.get("name", "yolov8n")
        default_weights = f"{name}.pt"
        weights = (
            str(checkpoint_path)
            if checkpoint_path
            else model_cfg.get("pretrained_weights", default_weights)
        )
        self.model = YOLO(weights)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def train(
        self,
        data_yaml: str | Path,
        output_dir: str | Path,
        **kwargs: Any,
    ) -> dict:
        """Fine-tune the model on a YOLO-format dataset.

        Args:
            data_yaml: Path to the dataset ``data.yaml``.
            output_dir: Directory where Ultralytics saves run artefacts.
            **kwargs: Training hyper-parameter overrides (``epochs``,
                ``batch``, ``lr0``, …).

        Returns:
            ``{"best_checkpoint": Path, "metrics": dict, "save_dir": str}``
        """
        params = self._training_params(kwargs)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = self.model.train(
            data=str(data_yaml),
            epochs=params.get("epochs", 100),
            batch=params.get("batch_size", 16),
            imgsz=params.get("image_size", 640),
            optimizer=params.get("optimizer", "AdamW"),
            lr0=params.get("lr", 0.001),
            weight_decay=params.get("weight_decay", 0.0005),
            warmup_epochs=params.get("warmup_epochs", 5),
            patience=params.get("patience", 20),
            device=self.device,
            project=str(output_dir.parent),
            name=output_dir.name,
            exist_ok=True,
            verbose=True,
        )

        best_ckpt = Path(results.save_dir) / "weights" / "best.pt"
        return {
            "best_checkpoint": best_ckpt,
            "metrics": results.results_dict if hasattr(results, "results_dict") else {},
            "save_dir": results.save_dir,
        }

    def predict(
        self,
        source: str | Path | np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45,
        **kwargs: Any,
    ) -> list[dict]:
        """Run inference.

        Args:
            source: Image path, directory, URL, or NumPy array (HWC, BGR).
            conf: Confidence threshold.
            iou: NMS IoU threshold.
            **kwargs: Extra args forwarded to ``YOLO.predict()``.

        Returns:
            List of per-image dicts with ``boxes``, ``scores``, ``class_ids``.
        """
        raw = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False,
            **kwargs,
        )
        return self._parse_results(raw)

    def export_onnx(
        self,
        output_path: str | Path,
        image_size: int = 640,
        dynamic: bool = True,
        **kwargs: Any,
    ) -> Path:
        """Export to ONNX.

        Args:
            output_path: Target ``.onnx`` file path.
            image_size: Export input resolution.
            dynamic: Use dynamic axes for batch and image size.
            **kwargs: Extra args forwarded to ``YOLO.export()``.

        Returns:
            Path to the exported ONNX file.
        """
        exported = self.model.export(
            format="onnx",
            imgsz=image_size,
            dynamic=dynamic,
            **kwargs,
        )
        src = Path(exported)
        dst = Path(output_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        return dst

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load weights from a saved ``.pt`` checkpoint.

        Args:
            checkpoint_path: Path to the ``.pt`` file.
        """
        from ultralytics import YOLO

        self.model = YOLO(str(checkpoint_path))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_results(raw_results: list) -> list[dict]:
        """Convert Ultralytics result objects to plain dicts.

        Args:
            raw_results: Output of ``YOLO.predict()``.

        Returns:
            List of ``{"boxes": np.ndarray, "scores": np.ndarray,
            "class_ids": np.ndarray}`` per image.
        """
        parsed = []
        for r in raw_results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                parsed.append({
                    "boxes": np.empty((0, 4)),
                    "scores": np.array([]),
                    "class_ids": np.array([], dtype=int),
                })
            else:
                parsed.append({
                    "boxes": boxes.xyxy.cpu().numpy(),
                    "scores": boxes.conf.cpu().numpy(),
                    "class_ids": boxes.cls.cpu().numpy().astype(int),
                })
        return parsed
