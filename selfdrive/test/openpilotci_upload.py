#!/usr/bin/env python3
import os
import sys
import subprocess


def upload_file(path, name):
  from azure.storage.blob import BlockBlobService  # pylint: disable=no-name-in-module, import-error
  sas_token = os.getenv("TOKEN", None)
  if sas_token is None:
    sas_token = subprocess.check_output("az storage container generate-sas --account-name commadataci --name openpilotci --https-only --permissions lrw --expiry $(date -u '+%Y-%m-%dT%H:%M:%SZ' -d '+1 hour') --auth-mode login --as-user --output tsv", shell=True).decode().strip("\n")
  service = BlockBlobService(account_name="commadataci", sas_token=sas_token)
  service.create_blob_from_path("openpilotci", name, path)
  return "https://commadataci.blob.core.windows.net/openpilotci/" + name

if __name__ == "__main__":
  for f in sys.argv[1:]:
    name = os.path.basename(f)
    url = upload_file(f, name)
    print(url)
