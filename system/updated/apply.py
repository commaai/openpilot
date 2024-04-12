#!/usr/bin/env python3
# applies the finalized updates from FINALIZED to their proper directory
import os
import pathlib

from openpilot.system.updated.common import USERDATA, FINALIZED, get_valid_flag


if __name__ == "__main__":
  if not get_valid_flag(FINALIZED):
    print("no valid update found in FINALIZED, skipping...")
    exit(0)

  for f in pathlib.Path(FINALIZED).glob("*"):
    if f.is_dir():
      apply_dir = os.path.join(USERDATA, f.stem)
      print(f"apply finalized update: {f} to {apply_dir}")
