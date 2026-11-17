#!/usr/bin/env python3
import os
import shutil
import threading
from openpilot.common.hardware.hw import Paths
from openpilot.common.params import Params
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

GDRIVE_ATTR_NAME = 'user.gdrive'
GDRIVE_ATTR_VALUE = b'1'
DASHCAM_FILES = ('fcamera.hevc', 'ecamera.hevc', 'dcamera.hevc')
CRITICAL_BYTES = 512 * 1024 * 1024  # at this point recording matters more than pending uploads


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


def has_pending_upload(d: str) -> bool:
  """Dashcam segment with a video file that has not reached Google Drive yet."""
  path = os.path.join(Paths.log_root(), d)
  try:
    names = os.listdir(path)
  except OSError:
    return False
  for name in names:
    if name in DASHCAM_FILES:
      try:
        if getxattr(os.path.join(path, name), GDRIVE_ATTR_NAME) != GDRIVE_ATTR_VALUE:
          return True
      except OSError:
        return True
  return False


def deleter_step() -> tuple[bool, str | None]:
  free_bytes = get_available_bytes(default=MIN_BYTES + 1)
  out_of_bytes = free_bytes < MIN_BYTES
  out_of_percent = get_available_percent(default=MIN_PERCENT + 1) < MIN_PERCENT
  out_of_space = out_of_percent or out_of_bytes
  if not out_of_space:
    return False, None

  protect_uploads = Params().get_bool("DashcamActive") and free_bytes >= CRITICAL_BYTES
  dirs = listdir_by_creation(Paths.log_root())
  preserved_dirs = get_preserved_segments(dirs)

  # remove the earliest directory we can; pending dashcam uploads are preserved last
  for delete_dir in sorted(dirs, key=lambda d: (d in DELETE_LAST, d in preserved_dirs,
                                                protect_uploads and has_pending_upload(d))):
    delete_path = os.path.join(Paths.log_root(), delete_dir)

    if any(name.endswith(".lock") for name in os.listdir(delete_path)):
      continue

    try:
      cloudlog.info(f"deleting {delete_path}")
      shutil.rmtree(delete_path)
      return True, delete_path
    except OSError:
      cloudlog.exception(f"issue deleting {delete_path}")
  return True, None


def deleter_thread(exit_event: threading.Event):
  while not exit_event.is_set():
    out_of_space, _ = deleter_step()
    exit_event.wait(.1 if out_of_space else 30)


def main():
  deleter_thread(threading.Event())


if __name__ == "__main__":
  main()
