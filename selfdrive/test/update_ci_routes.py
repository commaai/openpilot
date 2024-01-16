#!/usr/bin/env python3
import os
import re
import subprocess
import sys
from functools import lru_cache
from typing import Iterable, Optional

from azure.storage.blob import ContainerClient
from tqdm import tqdm

from openpilot.selfdrive.car.tests.routes import routes as test_car_models_routes
from openpilot.selfdrive.test.process_replay.test_processes import source_segments as replay_segments
from openpilot.tools.lib.openpilotci import (DATA_CI_ACCOUNT, DATA_CI_ACCOUNT_URL, OPENPILOT_CI_CONTAINER,
                                                  DATA_CI_CONTAINER, get_azure_credential, get_container_sas, upload_file)

DATA_PROD_ACCOUNT = "commadata2"
DATA_PROD_CONTAINER = "commadata2"

SOURCES = [
  (DATA_PROD_ACCOUNT, DATA_PROD_CONTAINER),
  (DATA_CI_ACCOUNT, DATA_CI_CONTAINER),
]


@lru_cache
def get_azure_keys():
  dest_container = ContainerClient(DATA_CI_ACCOUNT_URL, OPENPILOT_CI_CONTAINER, credential=get_azure_credential())
  dest_key = get_container_sas(DATA_CI_ACCOUNT, OPENPILOT_CI_CONTAINER)
  source_keys = [get_container_sas(*s) for s in SOURCES]
  return dest_container, dest_key, source_keys


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


def sync_to_ci_public(route: str) -> bool:
  dest_container, dest_key, source_keys = get_azure_keys()
  key_prefix = route.replace('|', '/')
  dongle_id = key_prefix.split('/')[0]

  if next(dest_container.list_blob_names(name_starts_with=key_prefix), None) is not None:
    return True

  print(f"Uploading {route}")
  for (source_account, source_bucket), source_key in zip(SOURCES, source_keys, strict=True):
    # assumes az login has been run
    print(f"Trying {source_account}/{source_bucket}")
    cmd = [
      "azcopy",
      "copy",
      f"https://{source_account}.blob.core.windows.net/{source_bucket}/{key_prefix}?{source_key}",
      f"https://{DATA_CI_ACCOUNT}.blob.core.windows.net/{OPENPILOT_CI_CONTAINER}/{dongle_id}?{dest_key}",
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
