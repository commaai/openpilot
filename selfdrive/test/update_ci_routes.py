#!/usr/bin/env python3
import subprocess
import sys
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Iterable, Optional

from azure.storage.blob import BlobServiceClient, ContainerClient, ContainerSasPermissions, generate_container_sas
from tqdm import tqdm

from openpilot.selfdrive.car.tests.routes import routes as test_car_models_routes
from openpilot.selfdrive.locationd.test.test_laikad import UBLOX_TEST_ROUTE, QCOM_TEST_ROUTE
from openpilot.selfdrive.test.process_replay.test_processes import source_segments as replay_segments
from openpilot.selfdrive.test.openpilotci import (BASE_URL, DATA_CI_ACCOUNT, DATA_CI_ACCOUNT_URL, DATA_CI_CONTAINER,
                                                  get_azure_credential)

DATA_PROD_ACCOUNT = "commadata2"
DATA_PROD_CONTAINER = "commadata2"

SOURCES = [
  (DATA_PROD_ACCOUNT, DATA_PROD_CONTAINER),
  (DATA_CI_ACCOUNT, DATA_CI_CONTAINER),
]


@lru_cache
def get_blob_service(account_name: str) -> BlobServiceClient:
  return BlobServiceClient(
    account_url=f"https://{account_name}.blob.core.windows.net",
    credential=get_azure_credential(),
  )


@lru_cache
def get_container_sas(account_name: str, container_name: str):
  start_time = datetime.utcnow()
  expiry_time = start_time + timedelta(hours=1)
  blob_service = get_blob_service(account_name)

  return generate_container_sas(
    account_name,
    container_name,
    user_delegation_key=blob_service.get_user_delegation_key(start_time, expiry_time),
    permission=ContainerSasPermissions(read=True, write=True, list=True),
    expiry=expiry_time,
  )


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
    f"{BASE_URL}/{destpath}?{dest_key}",
    "--recursive=false",
    "--overwrite=false",
  ] + [f"--exclude-pattern={p}" for p in exclude_patterns]
  subprocess.check_call(cmd)


@lru_cache
def get_container_client() -> ContainerClient:
  return ContainerClient(DATA_CI_ACCOUNT_URL, DATA_CI_CONTAINER, credential=get_azure_credential())


def sync_to_ci_public(route: str) -> bool:
  key_prefix = route.replace('|', '/')
  dongle_id = key_prefix.split('/')[0]

  container_client = get_container_client()
  if next(container_client.list_blob_names(name_starts_with=key_prefix), None) is not None:
    return True

  print(f"Uploading {route}")
  for source_account, source_bucket in SOURCES:
    # assumes az login has been run
    print(f"Trying {source_account}/{source_bucket}")
    cmd = [
      "azcopy",
      "copy",
      f"https://{source_account}.blob.core.windows.net/{source_bucket}/{key_prefix}",
      f"{BASE_URL}/{dongle_id}",
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
    to_sync.extend([UBLOX_TEST_ROUTE, QCOM_TEST_ROUTE])
    to_sync.extend([rt.route for rt in test_car_models_routes])
    to_sync.extend([s[1].rsplit('--', 1)[0] for s in replay_segments])

  for r in tqdm(to_sync):
    if not sync_to_ci_public(r):
      failed_routes.append(r)

  if len(failed_routes):
    print("failed routes:", failed_routes)
