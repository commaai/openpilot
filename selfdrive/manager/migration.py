
import os

from openpilot.common.swaglog import cloudlog
from openpilot.common.run import run_cmd


STAGING_ROOT = os.getenv("UPDATER_STAGING_ROOT", "/data/safe_staging")
OVERLAY_MERGED = os.path.join(STAGING_ROOT, "merged")


def dismount_overlay() -> None:
  # git updater had an overlay, remove it here
  if os.path.ismount(OVERLAY_MERGED):
    cloudlog.info("unmounting existing overlay")
    run_cmd(["sudo", "umount", "-l", OVERLAY_MERGED])


def migrate_openpilot():
  # system level migration for ensursing smooth transitions between openpilot versions
  dismount_overlay()
  run_cmd(["sudo", "rm", "-rf", STAGING_ROOT])
