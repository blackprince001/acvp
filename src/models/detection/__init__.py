"""Detection models package."""

from .base import BaseDetectionModel
from .registry import get_model, list_models
from .rtdetr import RTDETRDetectionModel
from .yolo import YOLODetectionModel
from .yolov5 import YOLOv5DetectionModel
from .yolov8 import YOLOv8DetectionModel
from .yolov9 import YOLOv9DetectionModel
from .yolov10 import YOLOv10DetectionModel
from .yolov11 import YOLO11DetectionModel

__all__ = [
  "BaseDetectionModel",
  "YOLODetectionModel",
  "YOLOv5DetectionModel",
  "YOLOv8DetectionModel",
  "YOLOv9DetectionModel",
  "YOLOv10DetectionModel",
  "YOLO11DetectionModel",
  "RTDETRDetectionModel",
  "get_model",
  "list_models",
]
