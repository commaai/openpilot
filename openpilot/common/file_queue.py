import os
from pathlib import Path
import re
import tempfile
import time


class FileQueue:
  """Best-effort, single-reader spool. Pending filenames reserve space before writes.

  Producers share no locks or open handles. Scan pending before ready so a file
  being published can be counted twice, but is not lost between the two scans.
  The cap is approximate under concurrent directory changes.
  """
  CAPACITY = 64 * 1024 * 1024
  PAGE_SIZE = os.sysconf('SC_PAGE_SIZE')

  def __init__(self, path: str, capacity: int = CAPACITY):
    self.capacity = capacity
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
      usage = 0
      for directory in (self.pending, self.ready):
        with os.scandir(directory) as entries:
          for entry in entries:
            if metadata := self._metadata(entry.name):
              usage += self._charge(metadata[1])
              if usage > self.capacity:
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

  def receive(self) -> bytes | None:
    if not self._batch:
      self._batch = sorted(self.ready.iterdir(), reverse=True)
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
    # Incomplete files are never delivered. Only reclaim reservations whose
    # producer has exited; a slow live writer may still be using its file.
    for path in self.pending.iterdir():
      if (metadata := self._metadata(path.name)) is None:
        continue
      try:
        os.kill(metadata[0], 0)
      except ProcessLookupError:
        path.unlink(missing_ok=True)
      except PermissionError:
        pass
    return None
