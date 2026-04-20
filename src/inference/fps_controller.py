"""Fixed-frame-rate inference controller.

Maintains a consistent output FPS by sleeping when faster than target and
signalling frame skips when slower. Tracks actual achieved FPS.
"""

from __future__ import annotations

import time
from collections import deque

from loguru import logger


class FixedFPSController:
  """Pace a processing loop to a target FPS.

  Usage::

      ctrl = FixedFPSController(target_fps=30)
      for frame in frames:
          if ctrl.should_skip():
              continue
          process(frame)
          ctrl.tick()
  """

  def __init__(self, target_fps: float | None = 30.0, window: int = 30):
    self.target_fps = target_fps
    self.frame_interval = 1.0 / target_fps if target_fps and target_fps > 0 else 0.0
    self._last_tick: float | None = None
    self._start_time: float | None = None
    self._frame_times: deque[float] = deque(maxlen=window)
    self._processed = 0
    self._skipped = 0

  def start(self):
    self._start_time = time.perf_counter()
    self._last_tick = self._start_time

  def should_skip(self) -> bool:
    """Return True if the caller is running behind schedule and should drop the frame.

    If no target FPS is set, never skip.
    """
    if self.frame_interval == 0.0 or self._last_tick is None:
      return False
    # If we're already past the deadline for the *next* frame, skip this one.
    now = time.perf_counter()
    deadline = self._last_tick + self.frame_interval
    if now > deadline + self.frame_interval:
      self._skipped += 1
      return True
    return False

  def tick(self):
    """Mark the end of frame processing; sleep if ahead of schedule."""
    now = time.perf_counter()
    if self._start_time is None:
      self.start()
      now = time.perf_counter()

    if self.frame_interval > 0.0 and self._last_tick is not None:
      deadline = self._last_tick + self.frame_interval
      sleep_for = deadline - now
      if sleep_for > 0:
        time.sleep(sleep_for)
        now = time.perf_counter()

    if self._last_tick is not None:
      self._frame_times.append(now - self._last_tick)

    self._last_tick = now
    self._processed += 1

  def actual_fps(self) -> float:
    if not self._frame_times:
      return 0.0
    mean_dt = sum(self._frame_times) / len(self._frame_times)
    return 1.0 / mean_dt if mean_dt > 0 else 0.0

  def stats(self) -> dict:
    elapsed = (time.perf_counter() - self._start_time) if self._start_time else 0.0
    return {
      "target_fps": self.target_fps,
      "actual_fps": round(self.actual_fps(), 2),
      "processed": self._processed,
      "skipped": self._skipped,
      "elapsed_s": round(elapsed, 2),
    }

  def log_summary(self):
    s = self.stats()
    logger.info(
      "FPS: target={} actual={:.2f} processed={} skipped={} elapsed={:.2f}s",
      s["target_fps"],
      s["actual_fps"],
      s["processed"],
      s["skipped"],
      s["elapsed_s"],
    )
