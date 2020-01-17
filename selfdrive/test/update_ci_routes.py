#!/usr/bin/env python3
import os
import subprocess
import tempfile
import shutil
from azure.storage.blob import BlockBlobService

from selfdrive.test.test_car_models import routes as test_car_models_routes, non_public_routes
from selfdrive.test.process_replay.test_processes import segments as replay_segments
from xx.chffr.lib import azureutil
from xx.chffr.lib.storage import upload_dir_serial, download_dir_tpe, key_prefix_exists
from xx.chffr.lib.storage import _DATA_ACCOUNT_PRODUCTION, _DATA_ACCOUNT_CI, _DATA_BUCKET_PRODUCTION, _DATA_BUCKET_CI

sas_token = os.getenv("TOKEN", None)
if sas_token is None:
  sas_token = subprocess.check_output("az storage container generate-sas --account-name commadataci --name openpilotci --https-only --permissions lrw --expiry $(date -u '+%Y-%m-%dT%H:%M:%SZ' -d '+1 hour') --auth-mode login --as-user --output tsv", shell=True).decode().strip("\n")
service = BlockBlobService(account_name=_DATA_ACCOUNT_CI, sas_token=sas_token)

def sync_to_ci_public(service, route):
  key_prefix = route.replace('|', '/')

  if next(azureutil.list_all_blobs(service, "openpilotci", prefix=key_prefix), None) is not None:
    return

  print("uploading", route)

  tmpdir = tempfile.mkdtemp()
  try:
    print(f"download_dir_tpe({_DATA_ACCOUNT_PRODUCTION}, {_DATA_BUCKET_PRODUCTION}, {key_prefix}, {tmpdir})")

    # production -> openpilotci
    download_dir_tpe(_DATA_ACCOUNT_PRODUCTION, _DATA_BUCKET_PRODUCTION, tmpdir, key_prefix)

    # commadataci -> openpilotci
    #download_dir_tpe(_DATA_ACCOUNT_CI, _DATA_BUCKET_CI, tmpdir, key_prefix)

    upload_dir_serial(_DATA_ACCOUNT_CI, "openpilotci", tmpdir, key_prefix)
  finally:
    shutil.rmtree(tmpdir)

# sync process replay routes
for s in replay_segments:
  route_name, _ = s.rsplit('--', 1)
  sync_to_ci_public(service, route_name)

# sync test_car_models routes
for r in test_car_models_routes:
  if r not in non_public_routes:
    sync_to_ci_public(service, r)
