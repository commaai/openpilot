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


def safe_delete(path: str) -> bool:
  """Safely delete a file or directory, handling various edge cases.

  Returns True if deletion was successful, False otherwise.
  """
  try:
    if os.path.islink(path):
      # Handle symlinks - remove the link, not the target
      os.unlink(path)
    elif os.path.isfile(path):
      # Handle regular files
      os.remove(path)
    elif os.path.isdir(path):
      # Handle directories
      shutil.rmtree(path)
    else:
      # Handle other types (sockets, FIFOs, etc.)
      os.remove(path)
    cloudlog.info(f"deleted {path}")
    return True
  except (OSError, PermissionError) as e:
    cloudlog.exception(f"failed to delete {path}: {e}")
    return False


def deleter_thread(exit_event: threading.Event):
  while not exit_event.is_set():
    out_of_bytes = get_available_bytes(default=MIN_BYTES + 1) < MIN_BYTES
    out_of_percent = get_available_percent(default=MIN_PERCENT + 1) < MIN_PERCENT

    if out_of_percent or out_of_bytes:
      log_root = Paths.log_root()

      # First, check for and delete any non-directory items (stray files, symlinks, etc.)
      try:
        all_items = os.listdir(log_root)
      except OSError:
        cloudlog.exception(f"failed to list {log_root}")
        exit_event.wait(.1)
        continue

      # Find non-directory items
      non_dirs = []
      for item in all_items:
        item_path = os.path.join(log_root, item)
        try:
          # Check for symlinks first (before isdir, which follows symlinks)
          if os.path.islink(item_path):
            non_dirs.append(item)
          elif not os.path.isdir(item_path):
            non_dirs.append(item)
        except OSError:
          # If we can't stat it, treat it as a non-directory to attempt deletion
          non_dirs.append(item)

      # Delete non-directory items first (stray files, symlinks, etc.)
      deleted_non_dir = False
      for item in non_dirs:
        item_path = os.path.join(log_root, item)
        if safe_delete(item_path):
          deleted_non_dir = True
          break

      # If we deleted a non-directory item, wait and continue to next iteration
      if deleted_non_dir:
        exit_event.wait(.1)
        continue

      # Get directories sorted by creation time (using the original function)
      dirs = listdir_by_creation(log_root)
      preserved_dirs = get_preserved_segments(dirs)

      # Remove the earliest directory we can
      for delete_dir in sorted(dirs, key=lambda d: (d in DELETE_LAST, d in preserved_dirs)):
        delete_path = os.path.join(log_root, delete_dir)

        try:
          # Check for lock files
          if any(name.endswith(".lock") for name in os.listdir(delete_path)):
            continue
        except OSError:
          # If we can't list the directory, try to delete it anyway
          pass

        if safe_delete(delete_path):
          break

      exit_event.wait(.1)
    else:
      exit_event.wait(30)


def main():
  deleter_thread(threading.Event())


if __name__ == "__main__":
  main()
