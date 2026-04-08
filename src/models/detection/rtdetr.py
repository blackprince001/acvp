"""RT-DETR detection model wrapper — thin alias over :class:`YOLODetectionModel`."""

from .yolo import YOLODetectionModel


class RTDETRDetectionModel(YOLODetectionModel):
    """RT-DETR detection wrapper (rtdetr-l / rtdetr-x).

    RT-DETR (Real-Time Detection Transformer) is a transformer-based detector
    available through Ultralytics. It does not use NMS, which can change
    inference behaviour slightly — the ``iou`` parameter has no effect.

    Inherits all functionality from :class:`YOLODetectionModel`.
    """
