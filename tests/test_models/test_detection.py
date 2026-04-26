"""
Unit tests for detection model wrappers.

Tests cover:
- Config loading and constructor.
- Output shapes and types of predict().
- Registry lookup and error handling.
- COCO → YOLO bbox conversion utility in the training script.

These tests are designed to run **without a GPU** and **without downloading
actual YOLO weights** by mocking the Ultralytics ``YOLO`` class.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Make the project root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))


def _make_yolo_result(n_boxes: int = 3) -> MagicMock:
  """Create a fake Ultralytics results object with *n_boxes* detections."""
  result = MagicMock()
  result.boxes = MagicMock()
  result.boxes.__len__ = MagicMock(return_value=n_boxes)
  result.boxes.xyxy.cpu().numpy.return_value = np.zeros((n_boxes, 4), dtype=np.float32)
  result.boxes.conf.cpu().numpy.return_value = np.ones(n_boxes, dtype=np.float32) * 0.9
  result.boxes.cls.cpu().numpy.return_value = np.zeros(n_boxes, dtype=np.float32)
  return result


def _make_empty_yolo_result() -> MagicMock:
  result = MagicMock()
  result.boxes = None
  return result


def _yolov8_config() -> dict:
  return {
    "name": "yolov8n",
    "task": "detect",
    "pretrained": True,
    "pretrained_weights": "yolov8n.pt",
    "training": {
      "epochs": 5,
      "batch_size": 4,
      "image_size": 640,
      "optimizer": "AdamW",
      "lr": 0.001,
      "weight_decay": 0.0005,
      "warmup_epochs": 1,
      "patience": 5,
    },
  }


def _yolo11_config() -> dict:
  cfg = _yolov8_config()
  cfg["name"] = "yolo11n"
  cfg["pretrained_weights"] = "yolo11n.pt"
  return cfg


def _load_train_script_module():
  """Dynamically load train_detection.py as a module."""
  script_path = PROJECT_ROOT / "scripts" / "train_detection.py"
  spec = importlib.util.spec_from_file_location("train_detection_script", script_path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


class TestBaseDetectionModelInterface:
  """Verify that the abstract interface cannot be instantiated directly."""

  def test_cannot_instantiate_base(self):
    from src.models.detection.base import BaseDetectionModel

    with pytest.raises(TypeError):
      BaseDetectionModel({})  # type: ignore[abstract]

  def test_repr_contains_model_name_and_device(self):
    """__repr__ must include model name and device."""
    mock_ultralytics = MagicMock()
    mock_ultralytics.YOLO.return_value = MagicMock()

    with patch.dict("sys.modules", {"ultralytics": mock_ultralytics}):
      from src.models.detection.yolov8 import YOLOv8DetectionModel

      model = YOLOv8DetectionModel(_yolov8_config(), device="cpu")
    assert "yolov8n" in repr(model)
    assert "cpu" in repr(model)


class TestYOLOv8DetectionModel:
  def test_predict_output_shape(self):
    from src.models.detection.yolov8 import YOLOv8DetectionModel

    fake_inner_yolo = MagicMock()
    fake_inner_yolo.predict.return_value = [_make_yolo_result(3)]

    mock_ultralytics = MagicMock()
    mock_ultralytics.YOLO.return_value = fake_inner_yolo

    with patch.dict("sys.modules", {"ultralytics": mock_ultralytics}):
      model = YOLOv8DetectionModel(_yolov8_config(), device="cpu")
      results = model.predict(source="dummy.jpg")

    assert len(results) == 1
    r = results[0]
    assert "boxes" in r and "scores" in r and "class_ids" in r
    assert r["boxes"].shape == (3, 4), f"Expected (3, 4), got {r['boxes'].shape}"
    assert r["scores"].shape == (3,)
    assert r["class_ids"].shape == (3,)

  def test_predict_empty_result(self):
    from src.models.detection.yolov8 import YOLOv8DetectionModel

    fake_inner_yolo = MagicMock()
    fake_inner_yolo.predict.return_value = [_make_empty_yolo_result()]

    mock_ultralytics = MagicMock()
    mock_ultralytics.YOLO.return_value = fake_inner_yolo

    with patch.dict("sys.modules", {"ultralytics": mock_ultralytics}):
      model = YOLOv8DetectionModel(_yolov8_config(), device="cpu")
      results = model.predict(source="dummy.jpg")

    assert results[0]["boxes"].shape == (0, 4)

  def test_predict_passes_conf_and_iou(self):
    from src.models.detection.yolov8 import YOLOv8DetectionModel

    fake_inner_yolo = MagicMock()
    fake_inner_yolo.predict.return_value = [_make_yolo_result(1)]

    mock_ultralytics = MagicMock()
    mock_ultralytics.YOLO.return_value = fake_inner_yolo

    with patch.dict("sys.modules", {"ultralytics": mock_ultralytics}):
      model = YOLOv8DetectionModel(_yolov8_config(), device="cpu")
      model.predict(source="x.jpg", conf=0.5, iou=0.3)

    call_kwargs = fake_inner_yolo.predict.call_args
    assert call_kwargs.kwargs["conf"] == 0.5
    assert call_kwargs.kwargs["iou"] == 0.3

  def test_parse_results_multi_box(self):
    from src.models.detection.yolov8 import YOLOv8DetectionModel

    raw = [_make_yolo_result(4)]
    out = YOLOv8DetectionModel._parse_results(raw)
    assert out[0]["boxes"].shape == (4, 4)
    assert out[0]["scores"].shape == (4,)
    assert out[0]["class_ids"].dtype == int

  def test_parse_results_none_boxes(self):
    from src.models.detection.yolov8 import YOLOv8DetectionModel

    out = YOLOv8DetectionModel._parse_results([_make_empty_yolo_result()])
    assert out[0]["boxes"].shape == (0, 4)


class TestYOLO11DetectionModel:
  def test_predict_output_shape(self):
    from src.models.detection.yolov11 import YOLO11DetectionModel

    fake_inner_yolo = MagicMock()
    fake_inner_yolo.predict.return_value = [_make_yolo_result(5)]

    mock_ultralytics = MagicMock()
    mock_ultralytics.YOLO.return_value = fake_inner_yolo

    with patch.dict("sys.modules", {"ultralytics": mock_ultralytics}):
      model = YOLO11DetectionModel(_yolo11_config(), device="cpu")
      results = model.predict(source="dummy.jpg")

    assert results[0]["boxes"].shape == (5, 4)

  def test_parse_results_empty(self):
    from src.models.detection.yolov11 import YOLO11DetectionModel

    out = YOLO11DetectionModel._parse_results([_make_empty_yolo_result()])
    assert out[0]["boxes"].shape == (0, 4)


class TestDetectionRegistry:
  def test_get_model_yolov8n(self):
    from src.models.detection.registry import get_model
    from src.models.detection.yolov8 import YOLOv8DetectionModel

    mock_ultralytics = MagicMock()
    mock_ultralytics.YOLO.return_value = MagicMock()

    with patch.dict("sys.modules", {"ultralytics": mock_ultralytics}):
      model = get_model("yolov8n", config=_yolov8_config(), device="cpu")

    assert isinstance(model, YOLOv8DetectionModel)

  def test_get_model_yolo11n(self):
    from src.models.detection.registry import get_model
    from src.models.detection.yolov11 import YOLO11DetectionModel

    mock_ultralytics = MagicMock()
    mock_ultralytics.YOLO.return_value = MagicMock()

    with patch.dict("sys.modules", {"ultralytics": mock_ultralytics}):
      model = get_model("yolo11n", config=_yolo11_config(), device="cpu")

    assert isinstance(model, YOLO11DetectionModel)

  def test_get_model_unknown_raises(self):
    from src.models.detection.registry import get_model

    with pytest.raises(ValueError, match="Unknown model name"):
      get_model("resnet50", config={})

  def test_list_models(self):
    from src.models.detection.registry import list_models

    models = list_models()
    assert "yolov8" in models
    assert "yolov11" in models

  def test_get_model_missing_config_file_raises(self, tmp_path):
    from src.models.detection.registry import get_model

    with pytest.raises(FileNotFoundError):
      get_model("yolov8n", config_path=tmp_path / "nonexistent.yaml")


class TestCocoBboxToYolo:
  """Verify the bbox coordinate conversion function from the training script."""

  @pytest.fixture(autouse=True)
  def _import_fn(self):
    mod = _load_train_script_module()
    self.fn = mod.coco_bbox_to_yolo

  def test_centre_conversion(self):
    # COCO [0, 0, 100, 100] in 200x200 image -> cx=0.25, cy=0.25, w=0.5, h=0.5
    cx, cy, nw, nh = self.fn([0, 0, 100, 100], 200, 200)
    assert abs(cx - 0.25) < 1e-6
    assert abs(cy - 0.25) < 1e-6
    assert abs(nw - 0.5) < 1e-6
    assert abs(nh - 0.5) < 1e-6

  def test_full_image_box(self):
    # Box covering the entire image -> centre at 0.5, size 1.0
    cx, cy, nw, nh = self.fn([0, 0, 640, 480], 640, 480)
    assert abs(cx - 0.5) < 1e-6
    assert abs(cy - 0.5) < 1e-6
    assert abs(nw - 1.0) < 1e-6
    assert abs(nh - 1.0) < 1e-6

  def test_small_box(self):
    # A small bbox at center of 100x100 image
    cx, cy, nw, nh = self.fn([49, 49, 2, 2], 100, 100)
    assert abs(cx - 0.5) < 1e-6
    assert abs(cy - 0.5) < 1e-6

  def test_values_clamped_to_01(self):
    # Oversized bbox (annotation error) -> values must stay in [0, 1]
    cx, cy, nw, nh = self.fn([0, 0, 700, 500], 640, 480)
    assert 0.0 <= cx <= 1.0
    assert 0.0 <= cy <= 1.0
    assert 0.0 <= nw <= 1.0
    assert 0.0 <= nh <= 1.0
