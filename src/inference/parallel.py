"""Parallel execution engine for the inference pipeline.

Splits the per-frame pipeline into stages connected by PipelineQueue
instances so slower stages (detection on GPU) can overlap with faster
ones (feature extraction, I/O).

The design mirrors ``InferenceOrchestrator.run`` but hands off between
stages via queues so each stage runs in its own thread. Frames are
tagged with a monotonic index so the writer stage can reorder them.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from loguru import logger

from src.utils.types import FilteredResult, TrackingResult

from .fps_controller import FixedFPSController
from .queue import PipelineQueue
from .video_io import VideoReader, VideoWriter, draw_overlay

_SENTINEL = object()


@dataclass
class FramePacket:
  frame_idx: int
  timestamp: float
  frame: np.ndarray
  tracking: TrackingResult | None = None
  filtered: FilteredResult | None = None
  features: np.ndarray | None = None
  prediction: Any | None = None
  road_mask: np.ndarray | None = None


class ParallelPipeline:
  """Run a multi-stage pipeline with each stage in its own thread.

  Stages are callables ``(packet: FramePacket) -> FramePacket``. The
  pipeline wires them in order using bounded queues with configurable
  overflow policies so slow consumers can drop old frames rather than
  back-pressure the reader.

  Args:
      stages: List of ``(name, callable)`` stage definitions.
      queue_size: Size of each inter-stage queue.
      overflow_policy: Overflow policy for all queues.
  """

  def __init__(
    self,
    stages: list[tuple[str, Callable[[FramePacket], FramePacket]]],
    queue_size: int = 8,
    overflow_policy: str = "drop_oldest",
  ):
    if not stages:
      raise ValueError("at least one stage required")
    self.stages = stages
    self.queues: list[PipelineQueue] = [
      PipelineQueue(
        maxsize=queue_size, overflow_policy=overflow_policy, name=f"q:{name}"
      )
      for name, _ in stages
    ]
    self._threads: list[threading.Thread] = []
    self._stop = threading.Event()
    self._errors: list[BaseException] = []

  def run(
    self,
    reader: VideoReader,
    writer: VideoWriter | None = None,
    target_fps: float | None = None,
    max_frames: int | None = None,
    on_packet: Callable[[FramePacket], None] | None = None,
  ) -> dict[str, Any]:
    """Drive the pipeline from *reader* to *writer*, returning stats."""
    fps_ctrl = FixedFPSController(target_fps=target_fps)
    fps_ctrl.start()

    self._threads = []
    for i, (name, fn) in enumerate(self.stages):
      in_q = self.queues[i - 1] if i > 0 else None
      out_q = self.queues[i]
      t = threading.Thread(
        target=self._stage_loop,
        args=(name, fn, in_q, out_q),
        daemon=True,
        name=f"stage:{name}",
      )
      t.start()
      self._threads.append(t)

    # Stage[i] reads from stage[i-1]'s output queue; stage[0] reads from
    # a dedicated ingress queue populated by the reader below.
    ingress = PipelineQueue(
      maxsize=self.queues[0]._q.maxsize,
      overflow_policy=self.queues[0]._policy,
      name="q:ingress",
    )
    self._ingress = ingress

    try:
      for frame_idx, timestamp, frame in reader.frames():
        if max_frames is not None and frame_idx >= max_frames:
          break
        if self._stop.is_set():
          break
        if fps_ctrl.should_skip():
          continue
        packet = FramePacket(frame_idx=frame_idx, timestamp=timestamp, frame=frame)
        ingress.put(packet, timeout=2.0)
        fps_ctrl.tick()

      # Signal end of stream to stage 0; it will propagate sentinels down.
      ingress.put(_SENTINEL, timeout=5.0)
    except BaseException as exc:  # propagate to consumer threads
      self._stop.set()
      self._errors.append(exc)
      raise
    finally:
      # Drain the final queue into writer / callback.
      out_q = self.queues[-1]
      while True:
        item = out_q.get(timeout=2.0)
        if item is None:
          if self._all_stages_finished():
            break
          continue
        if item is _SENTINEL:
          break
        if on_packet is not None:
          on_packet(item)
        if writer is not None:
          annotated = draw_overlay(
            item.frame,
            tracking=item.tracking,
            filtered=item.filtered,
            prediction=item.prediction,
            road_mask=item.road_mask,
          )
          writer.write(annotated)

      self._join_threads()
      fps_ctrl.log_summary()

    if self._errors:
      raise self._errors[0]

    return {
      "fps": fps_ctrl.stats(),
      "queues": [q.stats() for q in self.queues],
    }

  def _stage_loop(
    self,
    name: str,
    fn: Callable[[FramePacket], FramePacket],
    in_q: PipelineQueue | None,
    out_q: PipelineQueue,
  ) -> None:
    source = in_q if in_q is not None else getattr(self, "_ingress", None)
    if source is None:
      # ingress not yet assigned — spin briefly.
      while source is None and not self._stop.is_set():
        time.sleep(0.01)
        source = getattr(self, "_ingress", None)
      if source is None:
        return

    while not self._stop.is_set():
      item = source.get(timeout=1.0)
      if item is None:
        continue
      if item is _SENTINEL:
        out_q.put(_SENTINEL, timeout=2.0)
        return
      try:
        result = fn(item)
      except BaseException as exc:
        logger.exception("[{}] stage error on frame {}", name, item.frame_idx)
        self._errors.append(exc)
        self._stop.set()
        out_q.put(_SENTINEL, timeout=2.0)
        return
      if result is not None:
        out_q.put(result, timeout=2.0)

  def _all_stages_finished(self) -> bool:
    return all(not t.is_alive() for t in self._threads)

  def _join_threads(self) -> None:
    for t in self._threads:
      t.join(timeout=5.0)
    self._threads.clear()

  def stop(self) -> None:
    self._stop.set()
