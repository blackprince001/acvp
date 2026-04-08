"""YOLOv5 detection model wrapper — thin alias over :class:`YOLODetectionModel`."""

from .yolo import YOLODetectionModel


class YOLOv5DetectionModel(YOLODetectionModel):
    """YOLOv5 family detection wrapper (yolov5n/s/m/l/x).

    Inherits all functionality from :class:`YOLODetectionModel`.
    """
