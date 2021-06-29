#!/usr/bin/env python3
import os
import sys
import subprocess
from xx.chffr.lib import azureutil  # pylint: disable=import-error
from xx.chffr.lib.storage import _DATA_ACCOUNT_CI # pylint: disable=import-error

BASE_URL = "https://commadataci.blob.core.windows.net/openpilotci/"

TOKEN_PATH = "/data/azure_token"
DEST_KEY = azureutil.get_user_token(_DATA_ACCOUNT_CI, "openpilotci")

def get_url(route_name, segment_num, log_type="rlog"):
  ext = "hevc" if log_type in ["fcamera", "dcamera"] else "bz2"
  return BASE_URL + "%s/%s/%s.%s" % (route_name.replace("|", "/"), segment_num, log_type, ext)

def upload_route(path):
  r, n = path.rsplit("--", 1)
  destpath = f"{r}/{n}"
  cmd = [
    "azcopy",
    "copy",
    f"{path}/*",
    "https://{}.blob.core.windows.net/{}/{}?{}".format(_DATA_ACCOUNT_CI, "openpilotci", destpath, DEST_KEY),
    "--recursive=false",
    "--overwrite=false",
    "--exclude-pattern=*/dcamera.hevc",
  ]
  subprocess.check_call(cmd)

def upload_file(path, name):
  from azure.storage.blob import BlockBlobService

  sas_token = None
  if os.path.isfile(TOKEN_PATH):
    sas_token = open(TOKEN_PATH).read().strip()

  if sas_token is None:
    sas_token = subprocess.check_output("az storage container generate-sas --account-name commadataci --name openpilotci --https-only --permissions lrw \
                                         --expiry $(date -u '+%Y-%m-%dT%H:%M:%SZ' -d '+1 hour') --auth-mode login --as-user --output tsv", shell=True).decode().strip("\n")
  service = BlockBlobService(account_name="commadataci", sas_token=sas_token)
  service.create_blob_from_path("openpilotci", name, path)
  return "https://commadataci.blob.core.windows.net/openpilotci/" + name

if __name__ == "__main__":
  for f in sys.argv[1:]:
    name = os.path.basename(f)
    url = upload_file(f, name)
    print(url)
