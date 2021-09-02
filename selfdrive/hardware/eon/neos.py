#!/usr/bin/env python3
import hashlib
import json
import os


NEOSUPDATE_DIR = "/data/neoupdate"

RECOVERY_DEV = "/dev/block/bootdevice/by-name/recovery"
RECOVERY_COMMAND = "/cache/recovery/command"

# TODO: check storage space before downloading


def download_neos_update(manifest: str):

  os.makedirs(NEOSUPDATE_DIR, exist_ok=True)

  # TODO: check manifest validity
  with open(manifest) as f:
    mjson = json.loads(f.read())

  with open(RECOVERY_DEVICE) as f:
    rhash = hashlib.sha256(f.read()).hexdigest()

  print(mjson)


def perform_neos_update():

  # all done, reboot into recovery
  with open(RECOVERY_DEV, "wb") as f:
    ota_fn = ""
    f.write(f"--update_package={ota_fn}")
  os.system("service call power 16 i32 0 s16 recovery i32 1")


if __name__ == "__main__":
  download_neos_update("neos.json")
