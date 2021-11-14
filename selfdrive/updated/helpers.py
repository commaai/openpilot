import os
import subprocess
from pathlib import Path

# NOTE: this file cannot import anything that must be built
from common.basedir import BASEDIR

LOCK_FILE = os.getenv("UPDATER_LOCK_FILE", "/tmp/safe_staging_overlay.lock")
STAGING_ROOT = os.getenv("UPDATER_STAGING_ROOT", "/data/safe_staging")

NEOSUPDATE_DIR = os.getenv("UPDATER_NEOSUPDATE_DIR", "/data/neoupdate")

OVERLAY_UPPER = os.path.join(STAGING_ROOT, "upper")
OVERLAY_METADATA = os.path.join(STAGING_ROOT, "metadata")
OVERLAY_MERGED = os.path.join(STAGING_ROOT, "merged")

OLD_OPENPILOT = Path(os.path.join(STAGING_ROOT, "old_openpilot"))

FINALIZED = Path(os.path.join(STAGING_ROOT, "finalized"))
OVERLAY_INIT = Path(os.path.join(BASEDIR, ".overlay_init"))


def run(cmd: List[str], cwd: Optional[str] = None, low_priority: bool = False):
  if low_priority:
    cmd = ["nice", "-n", "19"] + cmd
  return subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.STDOUT, encoding='utf8')


def git_dir_modified() -> bool:
  git_dir_path = os.path.join(BASEDIR, ".git")
  new_files = run(["find", git_dir_path, "-newer", str(OVERLAY_INIT)])
  return len(new_files.splitlines()) > 0
