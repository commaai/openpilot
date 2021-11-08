#!/usr/bin/env python3
import sys

# NOTE: this file cannot import anything that must be built
from selfdrive.updated.helpers import OLD_OPENPILOT, OVERLAY_INIT, FINALIZED

def should_swap() -> bool:
  # Check to see if there's a valid update available. Conditions:
  #
  # 1. The BASEDIR init file has to exist, with a newer modtime than anything in
  #    the BASEDIR Git repo. This checks for local development work or the user
  #    switching branches/forks, which should not be overwritten.
  # 2. The FINALIZED consistent file has to exist, indicating there's an update
  #    that completed successfully and synced to disk.


  if OVERLAY_INIT.is_file():
    # TODO: check against git dir
    if FINALIZED.is_file():
      if OLD_OPENPILOT.is_dir():
        # TODO: restore backup? This means the updater didn't start after swapping
        print("openpilot backup found, not updating")
      else:
        return True

  return True

if __name__ == "__main__":
  sys.exit(0 if should_swap() else 1)
