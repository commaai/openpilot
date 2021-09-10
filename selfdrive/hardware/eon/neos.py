#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import requests

NEOSUPDATE_DIR = "/data/neoupdate"

RECOVERY_DEV = "/dev/block/bootdevice/by-name/recovery"
RECOVERY_COMMAND = "/cache/recovery/command"


def download_file(url: str, fn: str, sha256: str, display_name: str):
  # handle fully or partially downloaded file
  if os.path.isfile(fn):
    with open(fn, "rb") as f:
      if hashlib.sha256(f.read()).hexdigest() == sha256:
        print("already cached", url, fn)
        return

  h = hashlib.sha256()
  with open(fn, "ab+") as f:
    f.seek(0)
    while True:
      dat = f.read(8192)
      if not dat:
        break
      h.update(dat)

    headers = {"Range": f"bytes={f.tell()}-"}
    r = requests.get(url, stream=True, allow_redirects=True, headers=headers)

    total = int(r.headers['Content-Length'])
    if 'Content-Range' in r.headers:
      total = int(r.headers['Content-Range'].split('/')[-1])
    for chunk in r.iter_content(chunk_size=8192):
      f.write(chunk)
      h.update(chunk)
      print(f"Downloading {display_name}: {f.tell() / total * 100}")

  if h.hexdigest() != sha256:
    os.unlink(fn)
    raise Exception("downloaded update failed hash check")


def check_hash(fn: str, sha256: str, length: int = -1) -> bool:
  if not os.path.exists(fn):
    return False

  with open(fn, "rb") as f:
    dat = f.read(length)
  return hashlib.sha256(dat).hexdigest() == sha256


def download_neos_update(manifest_path: str):
  with open(manifest_path) as f:
    m = json.loads(f.read())

  os.makedirs(NEOSUPDATE_DIR, exist_ok=True)

  # handle recovery updates
  if not check_hash(RECOVERY_DEV, m['recovery_hash'], m['recovery_len']):
    recovery_fn = os.path.join(NEOSUPDATE_DIR, os.path.basename(m['recovery_url']))
    download_file(m['recovery_url'], recovery_fn, m['recovery_hash'], "recovery")

    # flash recovery
    with open(recovery_fn, "rb") as update, open(RECOVERY_DEV, "w+b") as recovery:
      while True:
        dat = update.read(4096)
        if len(dat) == 0:
          break
        recovery.write(dat)
    assert check_hash(RECOVERY_DEV, m['recovery_hash'], m['recovery_len']), "recovery flash corrupted"

  # download OTA update
  ota_fn = os.path.join(NEOSUPDATE_DIR, os.path.basename(m['ota_url']))
  download_file(m['ota_url'], ota_fn, m['ota_hash'], "system")


def verify_update_ready(manifest_path: str) -> bool:
  with open(manifest_path) as f:
    m = json.loads(f.read())

  ota_fn = os.path.join(NEOSUPDATE_DIR, os.path.basename(m['ota_url']))
  recovery_flashed = check_hash(RECOVERY_DEV, m['recovery_hash'], m['recovery_len'])
  return recovery_flashed and check_hash(ota_fn, m['ota_hash'])


def perform_ota_update(manifest_path: str):
  # reboot into recovery
  with open(manifest_path) as f:
    m = json.loads(f.read())
  ota_fn = os.path.join(NEOSUPDATE_DIR, os.path.basename(m['ota_url']))
  with open(RECOVERY_COMMAND, "wb") as rf:
    rf.write(bytes(f"--update_package={ota_fn}\n", encoding='utf-8'))
  os.system("service call power 16 i32 0 s16 recovery i32 1")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="NEOS update utility",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--swap", action="store_true", help="Peform update after downloading")
  parser.add_argument("--swap-if-ready", action="store_true", help="Perform update if already downloaded")
  parser.add_argument("manifest", help="Manifest json")
  args = parser.parse_args()

  if args.swap_if_ready:
    if verify_update_ready(args.manifest):
      perform_ota_update(args.manifest)
  else:
    download_neos_update(args.manifest)
    if args.swap:
      perform_ota_update(args.manifest)
