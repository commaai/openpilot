#!/usr/bin/env python3
import os
import psutil
import requests
import shutil
import time

from pathlib import Path

from openpilot.common.basedir import BASEDIR
from openpilot.common.run import run_cmd
from openpilot.common.params import Params
from openpilot.common.realtime import set_core_affinity
from openpilot.common.swaglog import cloudlog
from openpilot.system.updated.casync import casync
from openpilot.system.hardware import AGNOS
from openpilot.system.updated.common import get_consistent_flag, set_consistent_flag
from openpilot.system.version import BuildMetadata, get_build_metadata, build_metadata_from_dict

from openpilot.system.hardware.tici.agnos import flash_partition, get_target_slot_number



UPDATE_DELAY = int(os.environ.get("UPDATE_DELAY", 60))

API_HOST = os.getenv('API_HOST', 'https://api.commadotai.com')
CHANNELS_API_ROOT = "v1/openpilot/channels"

LOCK_FILE = os.getenv("UPDATER_LOCK_FILE", "/tmp/safe_staging_overlay.lock")
STAGING_ROOT = os.getenv("UPDATER_STAGING_ROOT", "/data/safe_staging")

CASYNC_STAGING = Path(STAGING_ROOT) / "casync"        # where the casync update is pulled
CASYNC_TMPDIR = Path(STAGING_ROOT) / "casync_tmp"     # working directory for casync temp files
FINALIZED = os.path.join(STAGING_ROOT, "finalized")


def get_remote_available_channels() -> list | None:
  try:
    return list(requests.get(f"{API_HOST}/{CHANNELS_API_ROOT}").json())
  except Exception:
    cloudlog.exception("fetching remote channels")
    return None


def get_remote_channel_data(channel) -> tuple[BuildMetadata | None, dict | None]:
  try:
    data = requests.get(f"{API_HOST}/{CHANNELS_API_ROOT}/{channel}").json()
    return build_metadata_from_dict(data["build_metadata"]), data["manifest"]
  except Exception:
    cloudlog.exception("fetching remote manifest failed")
    return None, None


def check_update_available(current_directory, other_metadata: BuildMetadata):
  build_metadata = get_build_metadata(current_directory)
  return build_metadata.channel != other_metadata.channel or \
         build_metadata.openpilot.git_commit != other_metadata.openpilot.git_commit


def extract_directory_helper(entry, cache_directory, directory):
  target = casync.parse_caibx(entry["casync"]["caibx"])

  cache_filename = os.path.join(CASYNC_TMPDIR, "cache.tar")
  tmp_filename = os.path.join(CASYNC_TMPDIR, "tmp.tar")

  cloudlog.info("building tarball update cache...")
  start = time.monotonic()
  sources = [('cache', casync.DirectoryTarChunkReader(cache_directory, cache_filename), casync.build_chunk_dict(target))]
  cloudlog.info(f"tarball cache creation completed in {time.monotonic() - start} seconds")

  sources += [('remote', casync.RemoteChunkReader(casync.get_default_store(entry["casync"]["caibx"])), casync.build_chunk_dict(target))]

  cloudlog.info(f"extracting {entry['casync']['caibx']} to {directory}")
  start = time.monotonic()
  stats = casync.extract_directory(target, sources, directory, tmp_filename)
  cloudlog.info(f"extraction completed in {time.monotonic() - start} seconds with {stats}")



# TODO: this can be removed after all devices have moved away from overlay based git updater
OVERLAY_MERGED = os.path.join(STAGING_ROOT, "merged")

def dismount_overlay() -> None:
  if os.path.ismount(OVERLAY_MERGED):
    cloudlog.info("unmounting existing overlay")
    run_cmd(["sudo", "umount", "-l", OVERLAY_MERGED])


def setup_dirs():
  dismount_overlay()
  run_cmd(["sudo", "rm", "-rf", STAGING_ROOT])
  if os.path.isdir(STAGING_ROOT):
    shutil.rmtree(STAGING_ROOT)

  Path(STAGING_ROOT).mkdir()
  CASYNC_TMPDIR.mkdir()
  CASYNC_STAGING.mkdir()


def download_update(manifest):
  cloudlog.info(f"downloading update from: {manifest}")

  for entry in manifest:
    if entry["type"] == "path_tarred" and entry["path"] == "/data/openpilot":
      extract_directory_helper(entry, BASEDIR, CASYNC_STAGING)

  if AGNOS:
    target_slot_number = get_target_slot_number()
    cloudlog.info(f"Target slot {target_slot_number}")

    # set target slot as unbootable
    os.system(f"abctl --set_unbootable {target_slot_number}")

    for entry in manifest:
      if entry["type"] == "partition":
        flash_partition(target_slot_number, entry, cloudlog)


def finalize_update():
  cloudlog.info("creating finalized version of the overlay")
  set_consistent_flag(FINALIZED, False)

  if os.path.exists(FINALIZED):
    shutil.rmtree(FINALIZED)
  shutil.copytree(CASYNC_STAGING, FINALIZED, symlinks=True)

  set_consistent_flag(FINALIZED, True)
  cloudlog.info("done finalizing overlay")


def main():
  # set io priority and schedule on system cpus
  set_core_affinity([0, 1, 2, 3])
  proc = psutil.Process()
  if psutil.LINUX:
    proc.ionice(psutil.IOPRIO_CLASS_BE, value=7)

  setup_dirs()

  params = Params()

  update_failed_count = 0

  while True:
    # check for updates
    build_metadata = get_build_metadata(BASEDIR)
    params.put("UpdaterCurrentDescription", build_metadata.ui_description)
    target_channel = build_metadata.channel
    update_ready = get_consistent_flag(FINALIZED)
    remote_build_metadata, remote_manifest = get_remote_channel_data(target_channel)

    if remote_build_metadata is not None and remote_manifest is not None:
      update_available = check_update_available(BASEDIR, remote_build_metadata)

      # if we have an update ready, check if that is up to date
      if update_ready and not check_update_available(FINALIZED, remote_build_metadata):
        update_available = False

      if update_available:
        try:
          download_update(remote_manifest)

          finalize_update()
          update_ready = get_consistent_flag(FINALIZED)
          update_failed_count = 0
        except Exception:
          update_failed_count += 1
          cloudlog.exception("exception while downloading ...")

    else:
      update_failed_count += 1

    time.sleep(UPDATE_DELAY)


if __name__ == "__main__":
  main()
