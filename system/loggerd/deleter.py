#!/usr/bin/env python3
import os
import shutil
import threading
from openpilot.system.hardware.hw import Paths
from openpilot.common.swaglog import cloudlog
from openpilot.system.loggerd.config import get_available_bytes, get_available_percent
from openpilot.system.loggerd.uploader import listdir_by_creation
from openpilot.system.loggerd.xattr_cache import getxattr

MIN_BYTES = 5 * 1024 * 1024 * 1024
MIN_PERCENT = 10

DELETE_LAST = ['boot', 'crash']

PRESERVE_ATTR_NAME = 'user.preserve'
PRESERVE_ATTR_VALUE = b'1'
PRESERVE_COUNT = 5

class Priority:
  PRESERVED = 1
  CRITICAL = 2


def has_preserve_xattr(d: str) -> bool:
  return getxattr(os.path.join(Paths.log_root(), d), PRESERVE_ATTR_NAME) == PRESERVE_ATTR_VALUE


def get_preserved_segments(dirs_by_creation: list[str]) -> list[str]:
  # skip deleting most recent N preserved segments (and their prior segment)
  preserved = []
  for n, d in enumerate(filter(has_preserve_xattr, reversed(dirs_by_creation))):
    if n == PRESERVE_COUNT:
      break
    date_str, _, seg_str = d.rpartition("--")

    # ignore non-segment directories
    if not date_str:
      continue
    try:
      seg_num = int(seg_str)
    except ValueError:
      continue

    # preserve segment and two prior
    for _seg_num in range(max(0, seg_num - 2), seg_num + 1):
      preserved.append(f"{date_str}--{_seg_num}")

  return preserved


def deleter_thread(exit_event: threading.Event):
  while not exit_event.is_set():
    out_of_bytes = get_available_bytes(default=MIN_BYTES + 1) < MIN_BYTES
    out_of_percent = get_available_percent(default=MIN_PERCENT + 1) < MIN_PERCENT
    if not (out_of_bytes or out_of_percent):
      exit_event.wait(30)
      continue

    dirs = listdir_by_creation(Paths.log_root())
    preserved_segments = get_preserved_segments(dirs)

    priority_map = dict()
    deletion_candidates = []

    for d in dirs:
      fs = os.listdir(os.path.join(Paths.log_root(), d))

      if any(f.endswith(".lock") for f in fs):
        continue

      if d in DELETE_LAST:
        priority = Priority.CRITICAL
      elif d in preserved_segments:
        priority = Priority.PRESERVED
      else:
        priority = 0

      for f in fs:
        fp = os.path.join(d, f)
        deletion_candidates.append(fp)
        priority_map[fp] = priority

    # sort by priority, and oldest to newest
    for to_delete in sorted(deletion_candidates, key=priority_map.get):
      delete_path = os.path.join(Paths.log_root(), to_delete)
      try:
        cloudlog.info(f"deleting {delete_path}")
        if os.path.isfile(delete_path):
          os.remove(delete_path)
        else:
          shutil.rmtree(delete_path)
        break
      except OSError:
        cloudlog.exception(f"issue deleting {delete_path}")
    exit_event.wait(.1)


def main():
  deleter_thread(threading.Event())


if __name__ == "__main__":
  main()
