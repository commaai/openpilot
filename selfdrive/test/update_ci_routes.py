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
  # (DATA_CI_ACCOUNT, DATA_CI_CONTAINER),
]


def strip_log_data(data: bytes) -> bytes:
  lr = LogReader.from_bytes(data)
  new_bytes = b''
  for msg in lr:
    if msg.which() in PRESERVE_SERVICES:
      new_bytes += msg.as_builder().to_bytes()

  return bz2.compress(new_bytes)


@lru_cache
def get_azure_containers():
  source_containers = []
  for source_account, source_bucket in SOURCES:
    source_containers.append(ContainerClient(f"https://{source_account}.blob.core.windows.net", source_bucket, credential=get_azure_credential()))
  dest_container = ContainerClient(DATA_CI_ACCOUNT_URL, DATA_CI_CONTAINER, credential=get_azure_credential())
  return source_containers, dest_container


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
  source_containers, dest_container = get_azure_containers()
  key_prefix = route.replace('|', '/')
  dongle_id = key_prefix.split('/')[0]

  if next(dest_container.list_blob_names(name_starts_with=key_prefix), None) is not None and route != 'ad5a3fa719bc2f83|2023-10-17--19-48-42':
    print('Already exists in dest container:', route)
    return True

  print(f"Downloading {route}")
  for source_container in source_containers:
    print(f'Trying {source_container.container_name}')
    # Get all blobs (rlogs) for this route, strip personally identifiable data, and upload to CI
    blobs = list(source_container.list_blob_names(name_starts_with=key_prefix))
    blobs = [b for b in blobs if not re.match(r'.*/dcamera.hevc', b)]
    print('blobs', blobs)
    if len(blobs):
      break
  else:
    print(f'No segments found in source container: {DATA_PROD_ACCOUNT}')
    return False

  fail = False
  for blob_name in tqdm(blobs):
    # don't overwrite existing blobs, skip if exists
    if dest_container.get_blob_client(blob_name).exists():
      print('Already exists in dest container:', blob_name)
      # TODO: did this fail before?
      continue

    # print('deleting', blob_name, 'from container', dest_container.container_name)
    # dest_container.delete_blob(blob_name)
    data = source_container.download_blob(blob_name).readall()
    if strip_data:
      data = strip_log_data(data)

    print(f'Uploading {blob_name} to {dest_container.container_name}')
    dest_container.upload_blob(blob_name, data)

  print("Failed" if fail else "Success")
  return not fail


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
