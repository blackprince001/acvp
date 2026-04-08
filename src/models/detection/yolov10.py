"""YOLOv10 detection model wrapper — thin alias over :class:`YOLODetectionModel`."""

from .yolo import YOLODetectionModel


class YOLOv10DetectionModel(YOLODetectionModel):
    """YOLOv10 family detection wrapper (yolov10n/s/m/l/x/b).

    Inherits all functionality from :class:`YOLODetectionModel`.
    """
