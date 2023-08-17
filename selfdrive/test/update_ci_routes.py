#!/usr/bin/env python3
from functools import lru_cache
import sys
import subprocess
from tqdm import tqdm
from azure.storage.blob import BlockBlobService  # pylint: disable=import-error

from selfdrive.car.tests.routes import routes as test_car_models_routes
from selfdrive.locationd.test.test_laikad import UBLOX_TEST_ROUTE, QCOM_TEST_ROUTE
from selfdrive.test.process_replay.test_processes import source_segments as replay_segments
from xx.chffr.lib import azureutil  # pylint: disable=import-error
from xx.chffr.lib.storage import _DATA_ACCOUNT_PRODUCTION, _DATA_ACCOUNT_CI, _DATA_BUCKET_PRODUCTION  # pylint: disable=import-error

SOURCES = [
  (_DATA_ACCOUNT_PRODUCTION, _DATA_BUCKET_PRODUCTION),
  (_DATA_ACCOUNT_CI, "commadataci"),
]


@lru_cache
def get_azure_keys():
  dest_key = azureutil.get_user_token(_DATA_ACCOUNT_CI, "openpilotci")
  source_keys = [azureutil.get_user_token(account, bucket) for account, bucket in SOURCES]
  service = BlockBlobService(_DATA_ACCOUNT_CI, sas_token=dest_key)
  return dest_key, source_keys, service


def upload_route(path, exclude_patterns=None):
  dest_key, _, _ = get_azure_keys()
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

def sync_to_ci_public(route):
  dest_key, source_keys, service = get_azure_keys()
  key_prefix = route.replace('|', '/')
  dongle_id = key_prefix.split('/')[0]

  if next(azureutil.list_all_blobs(service, "openpilotci", prefix=key_prefix), None) is not None:
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
