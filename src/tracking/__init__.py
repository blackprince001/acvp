"""
Tracking module for object tracking in video sequences.

Provides tracker implementations wrapping BoT-SORT and ByteTrack
via Ultralytics, with support for movement vector calculation
and redundant counting prevention.
"""

from .base import BaseTracker, Track, TrackingOutput
from .counting import (
  CountingLine,
  CountingResult,
  CountingZone,
  CrossingDirection,
  VehicleCounter,
)
from .ultralytics_tracker import UltralyticsTracker

__all__ = [
  "BaseTracker",
  "CountingLine",
  "CountingResult",
  "CountingZone",
  "CrossingDirection",
  "Track",
  "TrackingOutput",
  "UltralyticsTracker",
  "VehicleCounter",
]
