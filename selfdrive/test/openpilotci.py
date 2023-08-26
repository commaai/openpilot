#!/usr/bin/env python3
import os
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import IO, Union

DATA_CI_ACCOUNT = "commadataci"
DATA_CI_ACCOUNT_URL = f"https://{DATA_CI_ACCOUNT}.blob.core.windows.net"
DATA_CI_CONTAINER = "openpilotci"
BASE_URL = f"{DATA_CI_ACCOUNT_URL}/{DATA_CI_CONTAINER}/"

TOKEN_PATH = Path("/data/azure_token")


def get_url(route_name: str, segment_num, log_type="rlog") -> str:
  ext = "hevc" if log_type.endswith('camera') else "bz2"
  return BASE_URL + f"{route_name.replace('|', '/')}/{segment_num}/{log_type}.{ext}"


@lru_cache
def get_azure_credential():
  if "AZURE_TOKEN" in os.environ:
    return os.environ["AZURE_TOKEN"]
  elif TOKEN_PATH.is_file():
    return TOKEN_PATH.read_text().strip()
  else:
    from azure.identity import AzureCliCredential
    return AzureCliCredential()


@lru_cache
def get_container_sas(account_name: str, container_name: str):
  from azure.storage.blob import BlobServiceClient, ContainerSasPermissions, generate_container_sas
  start_time = datetime.utcnow()
  expiry_time = start_time + timedelta(hours=1)
  blob_service = BlobServiceClient(
    account_url=f"https://{account_name}.blob.core.windows.net",
    credential=get_azure_credential(),
  )
  return generate_container_sas(
    account_name,
    container_name,
    user_delegation_key=blob_service.get_user_delegation_key(start_time, expiry_time),
    permission=ContainerSasPermissions(read=True, write=True, list=True),
    expiry=expiry_time,
  )


def upload_bytes(data: Union[bytes, IO], blob_name: str) -> str:
  from azure.storage.blob import BlobClient
  blob = BlobClient(
    account_url=DATA_CI_ACCOUNT_URL,
    container_name=DATA_CI_CONTAINER,
    blob_name=blob_name,
    credential=get_azure_credential(),
  )
  blob.upload_blob(data)
  return BASE_URL + blob_name


def upload_file(path: Union[str, os.PathLike], blob_name: str) -> str:
  with open(path, "rb") as f:
    return upload_bytes(f, blob_name)
