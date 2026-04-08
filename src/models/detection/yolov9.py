"""YOLOv9 detection model wrapper — thin alias over :class:`YOLODetectionModel`."""

from .yolo import YOLODetectionModel


class YOLOv9DetectionModel(YOLODetectionModel):
  """YOLOv9 family detection wrapper (yolov9s/m/c/e).

  Inherits all functionality from :class:`YOLODetectionModel`.
  """
