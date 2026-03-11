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


def has_preserve_xattr(d: str) -> bool:
  return getxattr(os.path.join(Paths.log_root(), d), PRESERVE_ATTR_NAME) == PRESERVE_ATTR_VALUE


def get_preserved_segments(dirs_by_creation: list[str]) -> set[str]:
  # skip deleting most recent N preserved segments (and their prior segment)
  preserved = set()
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
      preserved.add(f"{date_str}--{_seg_num}")

  return preserved


def list_root_entries(root: str) -> list[str]:
  try:
    entries = os.listdir(root)
  except OSError:
    cloudlog.exception("list_root_entries failed")
    return []

  dirs_by_creation = [
    d for d in listdir_by_creation(root)
    if not os.path.islink(os.path.join(root, d))
  ]
  preserved_dirs = get_preserved_segments(dirs_by_creation)
  dir_order = {d: i for i, d in enumerate(dirs_by_creation)}

  return sorted(
    entries,
    key=lambda d: (d in DELETE_LAST, d in preserved_dirs, d in dir_order, dir_order.get(d, -1), d),
  )


def delete_root_entry(root: str, entry: str) -> bool:
  delete_path = os.path.join(root, entry)

  try:
    is_dir = os.path.isdir(delete_path) and not os.path.islink(delete_path)
    if is_dir and any(name.endswith(".lock") for name in os.listdir(delete_path)):
      return False

    cloudlog.info(f"deleting {delete_path}")
    if is_dir:
      shutil.rmtree(delete_path)
    else:
      os.remove(delete_path)
    return True
  except OSError:
    cloudlog.exception(f"issue deleting {delete_path}")
    return False


def deleter_thread(exit_event: threading.Event):
  while not exit_event.is_set():
    out_of_bytes = get_available_bytes(default=MIN_BYTES + 1) < MIN_BYTES
    out_of_percent = get_available_percent(default=MIN_PERCENT + 1) < MIN_PERCENT

    if out_of_percent or out_of_bytes:
      # remove the earliest root entry we can
      for entry in list_root_entries(Paths.log_root()):
        if delete_root_entry(Paths.log_root(), entry):
          break
      exit_event.wait(.1)
    else:
      exit_event.wait(30)


def main():
  deleter_thread(threading.Event())


if __name__ == "__main__":
  main()
