import os
import re
import time
import random
from pathlib import Path


class ShmQueue:
  """Best-effort, single-reader queue with a fixed pool of file slots."""
  SLOT_COUNT = 1024
  CLAIM_ATTEMPTS = 8
  MAX_MESSAGE_SIZE = 128 * 1024

  def __init__(self, path: str):
    self.slots = Path(path) / 'slots'
    self.pending = Path(path) / 'pending'
    self.ready = Path(path) / 'ready'
    for directory in (self.slots, self.pending, self.ready):
      directory.mkdir(parents=True, exist_ok=True)
    self._batch: list[Path] = []

  @staticmethod
  def _metadata(name: str) -> tuple[int, int, int] | None:
    match = re.fullmatch(r'[0-9]{20}-([0-9]+)-([0-9]+)-([0-9]+)', name)
    if match and int(match[1]) > 0:
      return int(match[1]), int(match[2]), int(match[3])
    return None

  def send(self, data: bytes) -> bool:
    if len(data) > self.MAX_MESSAGE_SIZE:
      return False
    space = os.statvfs(self.slots)
    if space.f_bavail * space.f_frsize < len(data) + 128 * 1024 * 1024:
      return False
    for _ in range(self.CLAIM_ATTEMPTS):
      slot = self.slots / str(random.randrange(self.SLOT_COUNT))
      stamp = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
      name = f'{stamp:020d}-{os.getpid()}-{len(data)}-{slot.name}'
      try:
        # Exclusive creation with the owner's identity present from the start.
        slot.symlink_to(name)
        break
      except FileExistsError:
        continue
    else:
      return False

    pending = self.pending / name
    try:
      with pending.open('xb') as file:
        file.write(data)
      os.rename(pending, self.ready / name)
      return True
    except OSError:
      pending.unlink(missing_ok=True)
      slot.unlink(missing_ok=True)
      raise

  def _load_batch(self):
    # A claim survives publication. Reclaim abandoned writes only after exit.
    for slot in self.slots.iterdir():
      try:
        name = os.readlink(slot)
      except FileNotFoundError:
        continue
      if (metadata := self._metadata(name)) is None:
        continue
      try:
        os.kill(metadata[0], 0)
      except ProcessLookupError:
        (self.pending / name).unlink(missing_ok=True)
        if not (self.ready / name).exists():
          self._release(slot, name)
          continue
      except PermissionError:
        pass
    self._batch = sorted(self.ready.iterdir(), reverse=True)

  @staticmethod
  def _release(slot: Path, name: str):
    try:
      if os.readlink(slot) == name:
        slot.unlink()
    except FileNotFoundError:
      pass

  def receive(self) -> bytes | None:
    if not self._batch:
      self._load_batch()
    while self._batch:
      path = self._batch.pop()
      if (metadata := self._metadata(path.name)) is None:
        continue
      try:
        data = path.read_bytes()
        # Release first so a reader crash cannot strand a live producer's slot.
        # A replay after restart must not release a newer record's claim.
        self._release(self.slots / str(metadata[2]), path.name)
        path.unlink()
      except FileNotFoundError:
        continue
      if len(data) == metadata[1] and data:
        return data
    return None
