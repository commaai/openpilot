#!/usr/bin/env python3
import tempfile
import shutil
import subprocess
from common.basedir import BASEDIR
from azure.storage.blob import BlockBlobService

from selfdrive.test.test_car_models import routes as test_car_models_routes, non_public_routes
from selfdrive.test.process_replay.test_processes import segments as replay_segments
from xx.chffr.lib import azureutil
from xx.chffr.lib.storage import upload_dir_serial, download_dir_tpe
from xx.chffr.lib.storage import _DATA_ACCOUNT_PRODUCTION, _DATA_ACCOUNT_CI, _DATA_BUCKET_PRODUCTION, _DATA_BUCKET_CI

SOURCES = [
  (_DATA_ACCOUNT_PRODUCTION, _DATA_BUCKET_PRODUCTION),
  (_DATA_ACCOUNT_PRODUCTION, "preserve"),
]

DEST_KEY = azureutil.get_user_token(_DATA_ACCOUNT_CI, "openpilotci")
SOURCE_KEYS = [azureutil.get_user_token(account, bucket) for account, bucket in SOURCES]
SERVICE = BlockBlobService(_DATA_ACCOUNT_CI, sas_token=DEST_KEY)


def sync_to_ci_public(route):
  print(f"Uploading {route}")
  key_prefix = route.replace('|', '/')

  if next(azureutil.list_all_blobs(SERVICE, "openpilotci", prefix=key_prefix), None) is not None:
    print("Already synced")
    return True

  for (source_account, source_bucket), source_key in zip(SOURCES, SOURCE_KEYS):
    print(f"Trying {source_account}/{source_bucket}")
    cmd = [
      f"{BASEDIR}/external/bin/azcopy",
      "copy",
      "https://{}.blob.core.windows.net/{}/{}?{}".format(source_account, source_bucket, key_prefix, source_key),
      "https://{}.blob.core.windows.net/{}?{}".format(_DATA_ACCOUNT_CI, "openpilotci", DEST_KEY),
      "--recursive=true",
      "--overwrite=false",
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

  # sync process replay routes
  for s in replay_segments:
    route_name, _ = s.rsplit('--', 1)
    if not sync_to_ci_public(route_name):
      failed_routes.append(route_name)

  # sync test_car_models routes
  for r in list(test_car_models_routes.keys()):
    if r not in non_public_routes:
      if not sync_to_ci_public(r):
        failed_routes.append(r)


  if len(failed_routes):
    print("failed routes:")
    print(failed_routes)
