#!/usr/bin/env python3
import os
import shutil
import threading
from openpilot.common.hardware.hw import Paths
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.selfdrived.alertmanager import set_offroad_alert
from openpilot.system.loggerd.config import get_available_bytes, get_available_percent
from openpilot.system.loggerd.uploader import listdir_by_creation
from openpilot.system.loggerd.xattr_cache import getxattr

MIN_BYTES = 5 * 1024 * 1024 * 1024
MIN_PERCENT = 10

DELETE_LAST = ['boot', 'crash']

PRESERVE_ATTR_NAME = 'user.preserve'
PRESERVE_ATTR_VALUE = b'1'
PRESERVE_COUNT = 5
CLEANUP_ALERT = "Offroad_StorageCleanupFailed"
_cleanup_alert_shown = False


def report_cleanup_failure(message: str) -> None:
  global _cleanup_alert_shown
  cloudlog.exception(message)
  if not _cleanup_alert_shown:
    try:
      set_offroad_alert(CLEANUP_ALERT, True)
      _cleanup_alert_shown = True
    except Exception:
      cloudlog.exception("unable to show storage cleanup alert")


def clear_cleanup_alert() -> None:
  global _cleanup_alert_shown
  if _cleanup_alert_shown:
    try:
      set_offroad_alert(CLEANUP_ALERT, False)
      _cleanup_alert_shown = False
    except Exception:
      cloudlog.exception("unable to clear storage cleanup alert")


def list_recording_dirs(root: str) -> tuple[list[str], set[str]]:
  dirs, preserved = [], set()
  for d in listdir_by_creation(root):
    try:
      if getxattr(os.path.join(root, d), PRESERVE_ATTR_NAME) == PRESERVE_ATTR_VALUE:
        preserved.add(d)
      dirs.append(d)
    except OSError:
      report_cleanup_failure(f"issue inspecting {os.path.join(root, d)}")
  return dirs, preserved


def get_preserved_segments(dirs_by_creation: list[str], preserved_recordings: set[str]) -> set[str]:
  # skip deleting most recent N preserved segments (and their prior segment)
  preserved = set()
  for n, d in enumerate(d for d in reversed(dirs_by_creation) if d in preserved_recordings):
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


def deleter_step() -> tuple[bool, str | None]:
  out_of_bytes = get_available_bytes(default=MIN_BYTES + 1) < MIN_BYTES
  out_of_percent = get_available_percent(default=MIN_PERCENT + 1) < MIN_PERCENT
  out_of_space = out_of_percent or out_of_bytes
  if not out_of_space:
    clear_cleanup_alert()
    return False, None

  dirs, preserved_recordings = list_recording_dirs(Paths.log_root())
  preserved_dirs = get_preserved_segments(dirs, preserved_recordings)

  # remove the earliest directory we can
  for delete_dir in sorted(dirs, key=lambda d: (d in DELETE_LAST, d in preserved_dirs)):
    delete_path = os.path.join(Paths.log_root(), delete_dir)

    try:
      if any(name.endswith(".lock") for name in os.listdir(delete_path)):
        continue
      cloudlog.info(f"deleting {delete_path}")
      shutil.rmtree(delete_path)
      return True, delete_path
    except OSError:
      report_cleanup_failure(f"issue deleting {delete_path}")
  return True, None


def deleter_thread(exit_event: threading.Event):
  while not exit_event.is_set():
    out_of_space, _ = deleter_step()
    exit_event.wait(.1 if out_of_space else 30)


def main():
  deleter_thread(threading.Event())


if __name__ == "__main__":
  main()
