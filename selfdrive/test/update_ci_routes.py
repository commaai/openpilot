#!/usr/bin/env python3
import re
import bz2
import subprocess
import sys
from functools import lru_cache
from typing import Iterable, Optional

import azure.core.exceptions
from azure.storage.blob import ContainerClient
from tqdm import tqdm

from openpilot.tools.lib.logreader import LogReader
from openpilot.selfdrive.car.tests.routes import routes as test_car_models_routes
from openpilot.selfdrive.test.process_replay.test_processes import source_segments as replay_segments
from openpilot.selfdrive.test.openpilotci import (DATA_CI_ACCOUNT, DATA_CI_ACCOUNT_URL, DATA_CI_CONTAINER,
                                                  get_azure_credential, get_container_sas)

PRESERVE_SERVICES = ['can', 'carParams', 'pandaStates', 'pandaStateDEPRECATED']

DATA_PROD_ACCOUNT = "commadata2"
DATA_PROD_CONTAINER = "commadata2"

# TODO: why is DATA_CI_ACCOUNT in this list? it will just copy from itself to itself...
SOURCES = [
  (DATA_PROD_ACCOUNT, DATA_PROD_CONTAINER),
  (DATA_CI_ACCOUNT, DATA_CI_CONTAINER),
]


def strip_log_data(data: bytes) -> bytes:
  lr = LogReader.from_bytes(data)
  new_bytes = b''
  for msg in lr:
    if msg.which() in PRESERVE_SERVICES:
      new_bytes += msg.as_builder().to_bytes()

  return bz2.compress(new_bytes)


@lru_cache
def get_azure_keys():
  source_container = ContainerClient(f"https://{DATA_PROD_ACCOUNT}.blob.core.windows.net", DATA_PROD_CONTAINER, credential=get_azure_credential())
  dest_container = ContainerClient(DATA_CI_ACCOUNT_URL, DATA_CI_CONTAINER, credential=get_azure_credential())
  dest_key = get_container_sas(DATA_CI_ACCOUNT, DATA_CI_CONTAINER)
  source_keys = [get_container_sas(*s) for s in SOURCES]
  return source_container, dest_container, dest_key, source_keys


def upload_route(path: str, exclude_patterns: Optional[Iterable[str]] = None) -> None:
  # TODO: use azure-storage-blob instead of azcopy, simplifies auth
  dest_key = get_container_sas(DATA_CI_ACCOUNT, DATA_CI_CONTAINER)
  if exclude_patterns is None:
    exclude_patterns = ['*/dcamera.hevc']

  r, n = path.rsplit("--", 1)
  r = '/'.join(r.split('/')[-2:])  # strip out anything extra in the path
  destpath = f"{r}/{n}"
  cmd = [
    "azcopy",
    "copy",
    f"{path}/*",
    f"https://{DATA_CI_ACCOUNT}.blob.core.windows.net/{DATA_CI_CONTAINER}/{destpath}?{dest_key}",
    "--recursive=false",
    "--overwrite=false",
  ] + [f"--exclude-pattern={p}" for p in exclude_patterns]
  subprocess.check_call(cmd)


def sync_to_ci_public(route: str, strip_data: bool = False) -> bool:
  source_container, dest_container, dest_key, source_keys = get_azure_keys()
  key_prefix = route.replace('|', '/')
  dongle_id = key_prefix.split('/')[0]

  if next(dest_container.list_blob_names(name_starts_with=key_prefix), None) is not None and route != 'ad5a3fa719bc2f83|2023-10-17--19-48-42':
    return True

  # Get all blobs (rlogs) for this route, strip personally identifiable data, and upload to CI
  print(f"Downloading {route}")
  blobs = list(source_container.list_blob_names(name_starts_with=key_prefix))[1:]
  blobs = [b for b in blobs if not re.match(r'.*/dcamera.hevc', b)]
  if not len(blobs):
    raise Exception(f'No segments found in source container: {DATA_PROD_ACCOUNT}')
  print('blobs', blobs)

  fail = False
  for blob_name in blobs:
    print('deleting', blob_name, 'from container', dest_container.container_name)
    dest_container.delete_blob(blob_name)
    # exit(0)
    data = source_container.download_blob(blob_name).readall()
    if strip_data:
      data = strip_log_data(data)

    try:
      dest_container.upload_blob(blob_name, data)
    except azure.core.exceptions.ResourceExistsError:
      print('Already exists in dest container:', blob_name)
      fail = True

  if fail:
    print("Failed")
  else:
    print("Success")
  return not fail
  # print(data)
  exit(1)

  print(f"Uploading {route}")
  for (source_account, source_bucket), source_key in zip(SOURCES, source_keys, strict=True):
    # assumes az login has been run
    print(f"Trying {source_account}/{source_bucket}")
    cmd = [
      "azcopy",
      "copy",
      f"https://{source_account}.blob.core.windows.net/{source_bucket}/{key_prefix}?{source_key}",
      f"https://{DATA_CI_ACCOUNT}.blob.core.windows.net/{DATA_CI_CONTAINER}/{dongle_id}?{dest_key}",
      "--recursive=true",
      "--overwrite=false",
      "--exclude-pattern=*/dcamera.hevc",
    ]

    try:
      result = subprocess.call(cmd, stdout=subprocess.DEVNULL)
      if result == 0:
        print("Success")
        return True
    except subprocess.CalledProcessError:
      print("Failed")

  return False


if __name__ == "__main__":
  failed_routes = []

  to_sync = sys.argv[1:]

  if not len(to_sync):
    # sync routes from the car tests routes and process replay
    to_sync.extend([rt.route for rt in test_car_models_routes])
    to_sync.extend([s[1].rsplit('--', 1)[0] for s in replay_segments])

  for r in tqdm(to_sync):
    if not sync_to_ci_public(r):
      failed_routes.append(r)

  if len(failed_routes):
    print("failed routes:", failed_routes)
