import queue
import threading
from typing import Any, Literal

from loguru import logger

OverflowPolicy = Literal["drop_oldest", "block", "drop_newest"]


class PipelineQueue:
  """Thread-safe queue with configurable overflow handling for inter-stage communication."""

  def __init__(
    self,
    maxsize: int = 10,
    overflow_policy: OverflowPolicy = "drop_oldest",
    name: str = "queue",
  ):
    if maxsize <= 0:
      raise ValueError("maxsize must be positive")
    if overflow_policy not in ("drop_oldest", "block", "drop_newest"):
      raise ValueError(f"unknown overflow_policy: {overflow_policy}")

    self._q: queue.Queue = queue.Queue(maxsize=maxsize)
    self._policy: OverflowPolicy = overflow_policy
    self._name = name
    self._lock = threading.Lock()
    self._dropped = 0
    self._put_count = 0
    self._get_count = 0
    self._closed = threading.Event()

  def put(self, item: Any, timeout: float = 1.0) -> bool:
    """Enqueue item. Returns True if stored, False if dropped."""
    if self._closed.is_set():
      return False

    if self._policy == "block":
      try:
        self._q.put(item, timeout=timeout)
      except queue.Full:
        logger.warning(f"[{self._name}] put timed out after {timeout}s")
        with self._lock:
          self._dropped += 1
        return False
      with self._lock:
        self._put_count += 1
      return True

    if self._policy == "drop_newest":
      try:
        self._q.put_nowait(item)
        with self._lock:
          self._put_count += 1
        return True
      except queue.Full:
        with self._lock:
          self._dropped += 1
        return False

    # drop_oldest
    while True:
      try:
        self._q.put_nowait(item)
        with self._lock:
          self._put_count += 1
        return True
      except queue.Full:
        try:
          self._q.get_nowait()
          with self._lock:
            self._dropped += 1
        except queue.Empty:
          continue

  def get(self, timeout: float = 1.0) -> Any | None:
    """Dequeue item. Returns None on timeout or when closed and empty."""
    try:
      item = self._q.get(timeout=timeout)
    except queue.Empty:
      return None
    with self._lock:
      self._get_count += 1
    return item

  def qsize(self) -> int:
    return self._q.qsize()

  def empty(self) -> bool:
    return self._q.empty()

  def close(self):
    """Signal that no more items will be put; consumers may still drain."""
    self._closed.set()

  def is_closed(self) -> bool:
    return self._closed.is_set()

  def stats(self) -> dict:
    with self._lock:
      return {
        "name": self._name,
        "size": self._q.qsize(),
        "put_count": self._put_count,
        "get_count": self._get_count,
        "dropped": self._dropped,
        "policy": self._policy,
      }
