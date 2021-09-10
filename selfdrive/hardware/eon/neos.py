#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import requests

NEOSUPDATE_DIR = "/data/neoupdate"

RECOVERY_DEV = "/dev/block/bootdevice/by-name/recovery"
RECOVERY_COMMAND = "/cache/recovery/command"

# TODO: check storage space before downloading


def download_file(url: str, fn: str, sha256: str, display_name: str):
  # TODO: handle file resuming
  h = hashlib.sha256()
  with open(fn, "wb") as f:
    r = requests.get(url, stream=True, allow_redirects=True)
    total = int(r.headers.get('Content-Length', 0))
    pos = 0
    for chunk in r.iter_content(chunk_size=8192):
      pos += f.write(chunk)
      h.update(chunk)
      print(f"Downloading {display_name}: {pos / total * 100}")
  assert h == sha256, "downloaded update failed hash check"

def check_recovery_hash(sha256, length):
  with open(RECOVERY_DEV, "rb") as f:
    dat = f.read(length)
  return hashlib.sha256(dat).hexdigest() == sha256

def handle_recovery_update(manifest: dict):
  recovery_url = manifest['recovery_url']
  recovery_len = manifest['recovery_len']
  recovery_hash = manifest['recovery_hash']

  # skip recovery install
  if check_recovery_hash(recovery_hash, recovery_len):
    return

  # download udpated recovery
  recovery_fn = os.path.join(NEOSUPDATE_DIR, os.path.basename(recovery_url))
  download_file(recovery_url, recovery_fn, recovery_hash, "recovery")

  # write recovery
  with open(recovery_fn, "rb") as update, open(RECOVERY_DEV, "w+b") as recovery:
    while True:
      dat = update.read(4096)
      if len(dat) == 0:
        break
      recovery.write(dat)

  assert check_recovery_hash(recovery_hash, recovery_len), "recovery flash corrupted"


def download_neos_update(manifest_path: str) -> str:
  # TODO: check manifest validity
  with open(manifest_path) as f:
    m = json.loads(f.read())

  os.makedirs(NEOSUPDATE_DIR, exist_ok=True)

  # handle recovery updates
  handle_recovery_update(m)

  # download OTA update
  ota_fn = os.path.join(NEOSUPDATE_DIR, os.path.basename(m['ota_url']))
  download_file(m['ota_url'], ota_fn, m['ota_hash'], "system")

  return ota_fn


def perform_ota_update(ota_fn: str):
  # reboot into recovery
  with open(RECOVERY_DEV, "wb") as f:
    f.write(bytes(f"--update_package={ota_fn}", encoding='utf-8'))
  os.system("service call power 16 i32 0 s16 recovery i32 1")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="NEOS update utility",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--swap", action="store_true", help="Peform update after downloading")
  parser.add_argument("manifest", help="Manifest json")
  args = parser.parse_args()

  fn = download_neos_update(args.manifest)
  #if args.swap:
  #  perform_ota_update(fn)
