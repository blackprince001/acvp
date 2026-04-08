"""
YOLOv8 detection model wrapper — thin alias over :class:`YOLODetectionModel`.

Kept for backwards compatibility. All logic lives in :mod:`.yolo`.
"""

from .yolo import YOLODetectionModel


class YOLOv8DetectionModel(YOLODetectionModel):
  """YOLOv8 family detection wrapper (yolov8n/s/m/l/x).

  Inherits all functionality from :class:`YOLODetectionModel`.
  See that class for full documentation.
  """
