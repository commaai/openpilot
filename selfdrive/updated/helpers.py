import os
from pathlib import Path

# NOTE: this file cannot import anything that must be built
# TODO: check if basedir is ok to include
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
