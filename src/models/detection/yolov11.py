"""
YOLO11 detection model wrapper — thin alias over :class:`YOLODetectionModel`.

Kept for backwards compatibility. All logic lives in :mod:`.yolo`.
"""

from .yolo import YOLODetectionModel


class YOLO11DetectionModel(YOLODetectionModel):
    """YOLO11 family detection wrapper (yolo11n/s/m/l/x).

    Inherits all functionality from :class:`YOLODetectionModel`.
    See that class for full documentation.
    """
