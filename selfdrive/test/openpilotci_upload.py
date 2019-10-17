#!/usr/bin/env python3
import os
import sys
import subprocess


def upload_file(path, name):
  from azure.storage.blob import BlockBlobService  # pylint: disable=no-name-in-module, import-error
  sas_token = os.getenv("TOKEN", None)
  if sas_token is not None:
    service = BlockBlobService(account_name="commadataci", sas_token=sas_token)
  else:
    account_key = subprocess.check_output("az storage account keys list --account-name commadataci --output tsv --query '[0].value'", shell=True)
    service = BlockBlobService(account_name="commadataci", account_key=account_key)
  service.create_blob_from_path("openpilotci", name, path)
  return "https://commadataci.blob.core.windows.net/openpilotci/" + name

if __name__ == "__main__":
  for f in sys.argv[1:]:
    name = os.path.basename(f)
    url = upload_file(f, name)
    print(url)
