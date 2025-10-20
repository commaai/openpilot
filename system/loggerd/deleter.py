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


def deleter_thread(exit_event: threading.Event):
  while not exit_event.is_set():
    out_of_bytes = get_available_bytes(default=MIN_BYTES + 1) < MIN_BYTES
    out_of_percent = get_available_percent(default=MIN_PERCENT + 1) < MIN_PERCENT

    if out_of_percent or out_of_bytes:
      dirs = listdir_by_creation(Paths.log_root())
      preserved_dirs = get_preserved_segments(dirs)

      # Get all items in log_root (both directories and files)
      log_root = Paths.log_root()
      try:
        all_items = os.listdir(log_root)
      except OSError:
        cloudlog.exception(f"failed to list {log_root}")
        exit_event.wait(.1)
        continue

      # Separate directories (from listdir_by_creation) and other items (files, symlinks)
      # Other items will be deleted after directories
      other_items = [item for item in all_items if item not in dirs]

      # Combine: directories first (sorted by age/preservation), then other items
      items_to_check = sorted(dirs, key=lambda d: (d in DELETE_LAST, d in preserved_dirs)) + sorted(other_items)

      # remove the earliest item we can
      for item_name in items_to_check:
        delete_path = os.path.join(log_root, item_name)

        # Check if path exists (handle race conditions)
        if not os.path.exists(delete_path) and not os.path.islink(delete_path):
          continue

        # For directories, check for lock files
        if os.path.isdir(delete_path) and not os.path.islink(delete_path):
          try:
            if any(name.endswith(".lock") for name in os.listdir(delete_path)):
              continue
          except (OSError, NotADirectoryError):
            # If we can't list it, try to delete it anyway
            pass

        # Delete the item (file, directory, or symlink)
        try:
          cloudlog.info(f"deleting {delete_path}")
          if os.path.islink(delete_path):
            # For symlinks, remove the link itself (don't follow)
            os.unlink(delete_path)
          elif os.path.isfile(delete_path):
            # Regular file
            os.remove(delete_path)
          elif os.path.isdir(delete_path):
            # Directory
            shutil.rmtree(delete_path)
          else:
            # Unknown type, try to remove as file
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
