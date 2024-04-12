#!/usr/bin/env python3
# applies the finalized updates from FINALIZED to their proper directory
import pathlib
import shutil

from openpilot.system.updated.common import USERDATA, FINALIZED, get_valid_flag, set_valid_flag


if __name__ == "__main__":
  if not get_valid_flag(FINALIZED):
    print("no valid update found in FINALIZED, skipping...")
    exit(0)

  finalized = pathlib.Path(FINALIZED)
  userdata = pathlib.Path(USERDATA)

  previous_update = userdata / "previous_update"

  previous_update.mkdir(exist_ok=True)

  for finalized_dir in finalized.glob("*"):
    if finalized_dir.is_dir():
      previous_dir = previous_update / finalized_dir.stem
      target_dir = userdata / finalized_dir.stem

      print(f"apply finalized update: {finalized_dir} to {target_dir} and putting old update in {previous_dir}")

      if previous_dir.exists():
        shutil.rmtree(previous_dir)

      target_dir.rename(previous_dir)
      finalized_dir.rename(target_dir)

  set_valid_flag(FINALIZED, False)
