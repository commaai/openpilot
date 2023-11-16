#!/usr/bin/env python3
import bz2
import os
import re
import sys
from functools import lru_cache
from typing import Iterable, Optional

from azure.storage.blob import ContainerClient
from tqdm import tqdm

from openpilot.tools.lib.logreader import LogReader
from openpilot.selfdrive.car.tests.routes import routes as test_car_models_routes
from openpilot.selfdrive.test.process_replay.test_processes import source_segments as replay_segments
from openpilot.selfdrive.test.openpilotci import (DATA_CI_ACCOUNT, DATA_CI_ACCOUNT_URL, OPENPILOT_CI_CONTAINER,
                                                  DATA_CI_CONTAINER, get_azure_credential, get_container_sas, upload_file)

PRESERVE_SERVICES = ["can", "carParams", "pandaStates", "pandaStateDEPRECATED"]
DATA_PROD_ACCOUNT = "commadata2"
DATA_PROD_CONTAINER = "commadata2"

SOURCES = [
  (DATA_PROD_ACCOUNT, DATA_PROD_CONTAINER),
  (DATA_CI_ACCOUNT, DATA_CI_CONTAINER),
]


def strip_log_data(data: bytes) -> bytes:
  lr = LogReader.from_bytes(data)
  new_bytes = b""
  for msg in lr:
    if msg.which() in PRESERVE_SERVICES:
      new_bytes += msg.as_builder().to_bytes()
  return bz2.compress(new_bytes)


@lru_cache
def get_azure_containers():
  source_containers = []
  for source_account, source_bucket in SOURCES:
    source_containers.append(ContainerClient(f"https://{source_account}.blob.core.windows.net", source_bucket, credential=get_azure_credential()))
  dest_container = ContainerClient(DATA_CI_ACCOUNT_URL, OPENPILOT_CI_CONTAINER, credential=get_azure_credential())
  source_keys = [get_container_sas(*s) for s in SOURCES]
  return source_containers, dest_container, source_keys


def upload_route(path: str, exclude_patterns: Optional[Iterable[str]] = None) -> None:
  if exclude_patterns is None:
    exclude_patterns = [r'dcamera\.hevc']

  r, n = path.rsplit("--", 1)
  r = '/'.join(r.split('/')[-2:])  # strip out anything extra in the path
  destpath = f"{r}/{n}"
  for file in os.listdir(path):
    if any(re.search(pattern, file) for pattern in exclude_patterns):
      continue
    upload_file(os.path.join(path, file), f"{destpath}/{file}")


def sync_to_ci_public(route: str, strip_data: bool = False) -> bool:
  source_containers, dest_container, source_keys = get_azure_containers()
  key_prefix = route.replace('|', '/')

  if next(dest_container.list_blob_names(name_starts_with=key_prefix), None) is not None:
    print("Already exists in dest container:", route)
    return True

  # Get all blobs (rlogs) for this route, strip personally identifiable data, and upload to CI
  print(f"Downloading {route}")
  source_key = None
  for source_container, source_key in zip(source_containers, source_keys, strict=True):
    print(f"Trying {source_container.url}")
    blobs = list(source_container.list_blob_names(name_starts_with=key_prefix))
    blobs = [b for b in blobs if not re.match(r".*/dcamera.hevc", b)]
    print(f"Found {len(blobs)} segments")
    if len(blobs):
      break
  else:
    print("No segments found in source containers")
    print("Failed")
    return False

  for blob_name in blobs:
    if strip_data and re.search(r"rlog|qlog", blob_name):
      print('downloading', blob_name)
      data = source_container.download_blob(blob_name).readall()
      data = strip_log_data(data)

      print(f"Uploading {blob_name} to {dest_container.url}")
      dest_container.upload_blob(blob_name, data)
    else:
      print('copying', blob_name)
      dest_blob_client = dest_container.get_blob_client(blob_name)
      print(source_container.get_blob_client(blob_name).url)
      dest_blob_client.start_copy_from_url(f"{source_container.get_blob_client(blob_name).url}?{source_key}")

  print("Success")
  return True


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
