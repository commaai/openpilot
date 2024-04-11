#!/usr/bin/env python3
import hashlib
import os
import psutil
import requests
import shutil
import time

from pathlib import Path

from openpilot.common.api import API_HOST
from openpilot.common.basedir import BASEDIR
from openpilot.common.run import run_cmd
from openpilot.common.params import Params
from openpilot.common.realtime import set_core_affinity
from openpilot.common.swaglog import cloudlog
from openpilot.system.hardware import HARDWARE
from openpilot.system.updated.casync import casync
from openpilot.system.updated.common import get_consistent_flag, set_consistent_flag
from openpilot.system.version import BuildMetadata, get_build_metadata, build_metadata_from_dict, is_git_repo


UPDATE_DELAY = int(os.environ.get("UPDATE_DELAY", 60))

CHANNELS_API_ROOT = "v1/openpilot/channels"

STAGING_ROOT = os.getenv("UPDATER_STAGING_ROOT", "/data/safe_staging")

FINALIZED = Path(STAGING_ROOT) / "finalized"        # where the casync update is pulled
CASYNC_TMPDIR = Path(STAGING_ROOT) / "casync_tmp"   # working directory for casync temp files

INIT_FILE = Path(BASEDIR) / ".overlay_init"

UPDATED_FORCE_WRITE = os.getenv("UPDATED_FORCE_WRITE") == "1"


def get_remote_available_channels() -> list | None:
  try:
    return list(requests.get(os.path.join(API_HOST, CHANNELS_API_ROOT)).json())
  except requests.exceptions.RequestException:
    cloudlog.exception("fetching remote channels")
    return None


def get_remote_channel_data(channel: str) -> tuple[BuildMetadata | None, dict | None]:
  try:
    data = requests.get(os.path.join(API_HOST, CHANNELS_API_ROOT, channel)).json()
    return build_metadata_from_dict(data["build_metadata"]), data["manifest"]
  except requests.exceptions.RequestException:
    cloudlog.exception("fetching remote manifest failed")
    return None, None


def check_update_available(current_directory, other_metadata: BuildMetadata) -> bool:
  build_metadata = get_build_metadata(current_directory)
  return is_git_repo(current_directory) or build_metadata != other_metadata


def create_remote_chunk_source(caibx_url, chunks):
  return ('remote', casync.RemoteChunkReader(casync.get_default_store(caibx_url)), casync.build_chunk_dict(chunks))


def get_partition_hash(path: str, partition_size: int, full_check: bool) -> str:
  if full_check:
    hasher = hashlib.sha256()
    pos, chunk_size = 0, 1024 * 1024

    with open(path, 'rb') as out:
      while pos < partition_size:
        n = min(chunk_size, partition_size - pos)
        hasher.update(out.read(n))
        pos += n

    return hasher.hexdigest().lower()
  else:
    with open(path, 'rb+') as out:
      out.seek(partition_size)
      return out.read(64).decode("utf-8")


def set_partition_hash(path: str, partition_size: int, new_hash: bytes):
  with open(path, 'rb+') as out:
    out.seek(partition_size)
    assert len(new_hash) == 64
    out.write(new_hash)

  os.sync()


def extract_directory_helper(entry, cache_directory, directory):
  caibx_url = entry["casync"]["caibx"]
  target = casync.parse_caibx(caibx_url)

  shutil.rmtree(directory)

  cache_filename = os.path.join(CASYNC_TMPDIR, "cache.tar")
  tmp_filename = os.path.join(CASYNC_TMPDIR, "tmp.tar")

  cloudlog.info("building tarball update cache...")
  start = time.monotonic()
  sources = [('cache', casync.DirectoryTarChunkReader(cache_directory, cache_filename), casync.build_chunk_dict(target))]
  cloudlog.info(f"tarball cache creation completed in {time.monotonic() - start} seconds")

  sources += [create_remote_chunk_source(caibx_url, target)]

  cloudlog.info(f"extracting {caibx_url} to {directory}")
  start = time.monotonic()
  stats = casync.extract_directory(target, sources, directory, tmp_filename)
  cloudlog.info(f"extraction completed in {time.monotonic() - start} seconds with {stats}")


def extract_partition_helper(entry):
  cloudlog.info(f"extracting partition: {entry}")

  caibx_url = entry["casync"]["caibx"]
  chunks = casync.parse_caibx(caibx_url)

  target_path = entry["path"]

  sources = []

  if entry["ab"]:
    assert entry["path"][-2:] == "_a"
    target_path = entry["path"].replace("_a", HARDWARE.get_target_ab_slot())
    seed_path = entry["path"].replace("_a", HARDWARE.get_current_ab_slot())
    sources += [('seed', casync.FileChunkReader(seed_path), casync.build_chunk_dict(chunks))]

  sources += [
    ('target', casync.FileChunkReader(target_path), casync.build_chunk_dict(chunks)),
    create_remote_chunk_source(caibx_url, chunks)
  ]

  current_hash = get_partition_hash(target_path, entry["size"], entry["full_check"])
  target_hash = entry['hash_raw'].lower()

  cloudlog.info(f"{current_hash=}, {target_hash=}")
  if current_hash == target_hash and not UPDATED_FORCE_WRITE:
    return

  # Clear hash before flashing in case we get interrupted
  full_check = entry['full_check']
  if not full_check:
    set_partition_hash(target_path, entry["size"], b'\x00' * 64)


  cloudlog.info(f"extracting {caibx_url} to {target_path}")
  start = time.monotonic()
  stats = casync.extract(chunks, sources, target_path)
  cloudlog.info(f"extraction completed in {time.monotonic() - start} seconds with {stats}")

  # Write hash after successful flash
  if not full_check:
    set_partition_hash(target_path, entry["size"], entry['hash_raw'].lower().encode())


def setup_updater():
  run_cmd(["sudo", "rm", "-rf", STAGING_ROOT])
  if os.path.isdir(STAGING_ROOT):
    shutil.rmtree(STAGING_ROOT)

  Path(STAGING_ROOT).mkdir()
  CASYNC_TMPDIR.mkdir()
  FINALIZED.mkdir()

  INIT_FILE.touch()


def download_update(manifest):
  cloudlog.info(f"downloading update from: {manifest}")

  set_consistent_flag(FINALIZED, False)

  HARDWARE.prepare_target_ab_slot()

  for entry in manifest:
    if entry["type"] == "path_tarred":
      extract_directory_helper(entry, BASEDIR, FINALIZED)

    if entry["type"] == "partition":
      extract_partition_helper(entry)



def main():
  # set io priority and schedule on system cpus
  set_core_affinity([0, 1, 2, 3])
  proc = psutil.Process()
  if psutil.LINUX:
    proc.ionice(psutil.IOPRIO_CLASS_BE, value=7)

  setup_updater()

  params = Params()

  update_failed_count = 0

  while True:
    # check for updates
    build_metadata = get_build_metadata(BASEDIR)

    params.put("UpdaterCurrentDescription", build_metadata.ui_description)
    target_channel = params.get("UpdaterTargetChannel", encoding="utf-8")
    if target_channel is None:
      target_channel = build_metadata.channel

    params.put("UpdaterTargetChannel", target_channel)

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
          set_consistent_flag(FINALIZED, True)
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
