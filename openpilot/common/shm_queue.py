import os
import re
import time
import tempfile
from pathlib import Path


class ShmQueue:
  """Best-effort, single-reader queue using files in shared memory."""
  CAPACITY = 64 * 1024 * 1024
  PAGE_SIZE = os.sysconf('SC_PAGE_SIZE')

  def __init__(self, path: str, capacity: int = CAPACITY):
    self.capacity = capacity
    self.budget = Path(path) / 'budget'
    self.pending = Path(path) / 'pending'
    self.ready = Path(path) / 'ready'
    self.pending.mkdir(parents=True, exist_ok=True)
    self.ready.mkdir(exist_ok=True)
    self._batch: list[Path] = []

  @staticmethod
  def _metadata(name: str) -> tuple[int, int] | None:
    match = re.fullmatch(r'[0-9]{20}-([0-9]+)-([0-9]+)-[A-Za-z0-9_]+', name)
    if match and int(match[1]) > 0:
      return int(match[1]), int(match[2])
    return None

  @classmethod
  def _charge(cls, size: int) -> int:
    # Allow for file metadata and page rounding, including very small logs.
    return (1 + (size + cls.PAGE_SIZE - 1) // cls.PAGE_SIZE) * cls.PAGE_SIZE

  def send(self, data: bytes) -> bool:
    if self._charge(len(data)) > self.capacity:
      return False
    stamp = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    fd, name = tempfile.mkstemp(prefix=f'{stamp:020d}-{os.getpid()}-{len(data)}-', dir=self.pending)
    try:
      if not self._reserve(len(data)):
        return False
      view = memoryview(data)
      while view:
        written = os.write(fd, view)
        if written == 0:
          return False
        view = view[written:]
      os.close(fd)
      fd = -1
      os.rename(name, self.ready / Path(name).name)
      return True
    finally:
      if fd >= 0:
        os.close(fd)
      Path(name).unlink(missing_ok=True)

  def _reserve(self, size: int) -> bool:
    # One appended byte claims one page. Each sender has its own file offset,
    # so the end of its append tells it whether it exceeded the shared budget.
    pages = self._charge(size) // self.PAGE_SIZE
    limit = self.capacity // self.PAGE_SIZE
    fd = os.open(self.budget, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
      if os.fstat(fd).st_size + pages > limit:
        return False
      return os.write(fd, bytes(pages)) == pages and os.lseek(fd, 0, os.SEEK_CUR) <= limit
    finally:
      os.close(fd)

  def _load_batch(self):
    used = 0
    ready = []
    # Pending must be scanned first: publication may be counted twice, not missed.
    # Every producer creates its reservation BEFORE opening the budget file.
    for directory in (self.pending, self.ready):
      for path in directory.iterdir():
        if (metadata := self._metadata(path.name)) is None:
          continue
        if directory == self.pending:
          try:
            os.kill(metadata[0], 0)
          except ProcessLookupError:
            path.unlink(missing_ok=True)
            continue
          except PermissionError:
            pass
        else:
          ready.append(path)
        used += self._charge(metadata[1])

    # Replace rather than truncate: in-flight appenders retain their old budget.
    # Late reservations can overshoot by roughly one budget during this handoff.
    # A fixed staging name also bounds leftovers if the single reader crashes.
    staging = self.budget.with_name('budget.next')
    fd = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_CLOEXEC, 0o600)
    try:
      os.ftruncate(fd, min(used, self.capacity) // self.PAGE_SIZE)
      os.replace(staging, self.budget)
    finally:
      os.close(fd)
      staging.unlink(missing_ok=True)
    self._batch = sorted(ready, reverse=True)

  def receive(self) -> bytes | None:
    if not self._batch:
      self._load_batch()
    while self._batch:
      path = self._batch.pop()
      if (metadata := self._metadata(path.name)) is None:
        continue
      try:
        data = path.read_bytes()
        path.unlink()
      except FileNotFoundError:
        continue
      if len(data) == metadata[1] and data:
        return data
    return None
