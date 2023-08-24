#!/usr/bin/env python3
import subprocess
import sys
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Iterable, Optional, Tuple

from azure.identity import AzureCliCredential
from azure.storage.blob import (BlobServiceClient, ContainerClient, ContainerSasPermissions,
                                generate_container_sas, UserDelegationKey)
from tqdm import tqdm

from openpilot.selfdrive.car.tests.routes import routes as test_car_models_routes
from openpilot.selfdrive.locationd.test.test_laikad import UBLOX_TEST_ROUTE, QCOM_TEST_ROUTE
from openpilot.selfdrive.test.process_replay.test_processes import source_segments as replay_segments

_DATA_ACCOUNT_PRODUCTION = "commadata2"
_DATA_ACCOUNT_CI = "commadataci"
_DATA_BUCKET_PRODUCTION = "commadata2"

SOURCES = [
  (_DATA_ACCOUNT_PRODUCTION, _DATA_BUCKET_PRODUCTION),
  (_DATA_ACCOUNT_CI, "commadataci"),
]


@lru_cache
def get_blob_service(account_name: str) -> BlobServiceClient:
  account_url = f"https://{account_name}.blob.core.windows.net"
  return BlobServiceClient(account_url, credential=AzureCliCredential())


@lru_cache
def get_user_delegation_key(account_name: str) -> UserDelegationKey:
  start_time = datetime.utcnow()
  expiry_time = start_time + timedelta(hours=1)
  return get_blob_service(account_name).get_user_delegation_key(start_time, expiry_time)


@lru_cache
def get_container_sas(account_name: str, container_name: str):
  return generate_container_sas(
    account_name,
    container_name,
    user_delegation_key=get_user_delegation_key(account_name),
    permission=ContainerSasPermissions(read=True, write=True, list=True),
    expiry=datetime.utcnow() + timedelta(hours=1),
  )


@lru_cache
def get_azure_keys() -> Tuple[str, Iterable[Tuple[str, str]], ContainerClient]:
  dest_key = get_container_sas(_DATA_ACCOUNT_CI, "openpilotci")
  source_keys = [get_container_sas(account, bucket) for account, bucket in SOURCES]
  container = get_blob_service(_DATA_ACCOUNT_CI).get_container_client("openpilotci")
  return dest_key, source_keys, container


def upload_route(path: str, exclude_patterns: Optional[Iterable[str]] = None) -> None:
  dest_key = get_container_sas(_DATA_ACCOUNT_CI, "openpilotci")
  if exclude_patterns is None:
    exclude_patterns = ['*/dcamera.hevc']

  r, n = path.rsplit("--", 1)
  r = '/'.join(r.split('/')[-2:])  # strip out anything extra in the path
  destpath = f"{r}/{n}"
  cmd = [
    "azcopy",
    "copy",
    f"{path}/*",
    f"https://{_DATA_ACCOUNT_CI}.blob.core.windows.net/openpilotci/{destpath}?{dest_key}",
    "--recursive=false",
    "--overwrite=false",
  ] + [f"--exclude-pattern={p}" for p in exclude_patterns]
  subprocess.check_call(cmd)


def sync_to_ci_public(route: str) -> bool:
  dest_key, source_keys, container = get_azure_keys()
  key_prefix = route.replace('|', '/')
  dongle_id = key_prefix.split('/')[0]

  if next(container.list_blob_names(name_starts_with=key_prefix), None) is not None:
    return True

  print(f"Uploading {route}")
  for (source_account, source_bucket), source_key in zip(SOURCES, source_keys, strict=True):
    print(f"Trying {source_account}/{source_bucket}")
    cmd = [
      "azcopy",
      "copy",
      f"https://{source_account}.blob.core.windows.net/{source_bucket}/{key_prefix}?{source_key}",
      f"https://{_DATA_ACCOUNT_CI}.blob.core.windows.net/openpilotci/{dongle_id}?{dest_key}",
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
