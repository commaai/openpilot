#!/usr/bin/env python3
import os
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

DASHCAM_FILES = {'qlog', 'qcamera.ts'}
TRAINING_FILES = {'rlog', 'dcamera.hevc', 'ecamera.hevc', 'fcamera.hevc'}
TRAINING_MAX_BYTES = 5 * 1024 * 1024 * 1024

class Priority:
  CRITICAL = 4
  PRESERVED = 2
  NORMAL = 1
  LOWEST = 0


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


def get_training_segments(dirs_by_creation: list[str]) -> list[str]:
  training = []
  training_bytes = 0
  for d in reversed(dirs_by_creation):
    training_files = [f for f in os.listdir(os.path.join(Paths.log_root(), d)) if f in TRAINING_FILES]
    if not training_files:
      continue
    training_bytes += sum(os.stat(os.path.join(Paths.log_root(), d, f)).st_size for f in training_files)
    training.append(d)
    if training_bytes >= TRAINING_MAX_BYTES:
      break
  return training


def deleter_thread(exit_event: threading.Event):
  while not exit_event.is_set():
    out_of_bytes = get_available_bytes(default=MIN_BYTES + 1) < MIN_BYTES
    out_of_percent = get_available_percent(default=MIN_PERCENT + 1) < MIN_PERCENT

    if out_of_percent or out_of_bytes:
      dirs = listdir_by_creation(Paths.log_root())
      preserved_segments = get_preserved_segments(dirs)
      training_segments = get_training_segments(dirs)

      priority_map = dict()
      candidates = []

      for delete_dir in dirs:
        fns = os.listdir(os.path.join(Paths.log_root(), delete_dir))
        if any(name.endswith(".lock") for name in fns):
          continue
        elif not fns:
          try:
            os.rmdir(os.path.join(Paths.log_root(), delete_dir))
          except OSError:
            cloudlog.exception(f"issue deleting empty {delete_dir}")
          continue

        if delete_dir in DELETE_LAST:
          priority = Priority.CRITICAL
        elif delete_dir in preserved_segments:
          priority = Priority.PRESERVED
        else:
          priority = Priority.LOWEST

        for fn in fns:
          if fn in DASHCAM_FILES:
            priority += Priority.NORMAL
          elif fn in TRAINING_FILES and delete_dir in training_segments:
            priority += Priority.NORMAL

          fp = os.path.join(delete_dir, fn)
          priority_map[fp] = priority
          candidates.append(fp)

      # sort by priority, and oldest to newest
      for candidate in sorted(candidates, key=priority_map.get):
        delete_path = os.path.join(Paths.log_root(), candidate)
        try:
          cloudlog.info(f"deleting {delete_path}")
          os.remove(delete_path)
          break
        except OSError:
          cloudlog.exception(f"issue deleting {delete_path}")
      exit_event.wait(.1)
    else:
      exit_event.wait(30)


def main():
  deleter_thread(threading.Event())


if __name__ == "__main__":
  main()
