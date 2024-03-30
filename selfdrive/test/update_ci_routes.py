#!/usr/bin/env python3
import os
import re
import subprocess
import sys
from collections.abc import Iterable

from tqdm import tqdm

from openpilot.selfdrive.car.tests.routes import routes as test_car_models_routes
from openpilot.selfdrive.test.process_replay.test_processes import source_segments as replay_segments
from openpilot.tools.lib.azure_container import AzureContainer
from openpilot.tools.lib.openpilotcontainers import DataCIContainer, DataProdContainer, OpenpilotCIContainer

SOURCES: list[AzureContainer] = [
  DataProdContainer,
  DataCIContainer
]

DEST = OpenpilotCIContainer

def upload_route(path: str, exclude_patterns: Iterable[str] = None) -> None:
  if exclude_patterns is None:
    exclude_patterns = [r'dcamera\.hevc']

  r, n = path.rsplit("--", 1)
  r = '/'.join(r.split('/')[-2:])  # strip out anything extra in the path
  destpath = f"{r}/{n}"
  for file in os.listdir(path):
    if any(re.search(pattern, file) for pattern in exclude_patterns):
      continue
    DEST.upload_file(os.path.join(path, file), f"{destpath}/{file}")


def sync_to_ci_public(route: str) -> bool:
  dest_container, dest_key = DEST.get_client_and_key()
  key_prefix = route.replace('|', '/')
  dongle_id = key_prefix.split('/')[0]

  if next(dest_container.list_blob_names(name_starts_with=key_prefix), None) is not None:
    return True

  print(f"Uploading {route}")
  for source_container in SOURCES:
    # assumes az login has been run
    print(f"Trying {source_container.ACCOUNT}/{source_container.CONTAINER}")
    _, source_key = source_container.get_client_and_key()
    cmd = [
      "azcopy",
      "copy",
      f"{source_container.BASE_URL}{key_prefix}?{source_key}",
      f"{DEST.BASE_URL}{dongle_id}?{dest_key}",
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
