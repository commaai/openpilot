#!/usr/bin/env python3
import os
from pathlib import Path
from typing import IO, Union

DATA_CI_ACCOUNT = "commadataci"
DATA_CI_ACCOUNT_URL = f"https://{DATA_CI_ACCOUNT}.blob.core.windows.net"
DATA_CI_CONTAINER = "openpilotci"

TOKEN_PATH = Path("/data/azure_token")


def get_ci_blob_url(blob_name: str) -> str:
  return f"{DATA_CI_ACCOUNT_URL}/{DATA_CI_CONTAINER}/{blob_name}"


def get_url(route_name: str, segment_num: str, log_type="rlog") -> str:
  ext = "hevc" if log_type.endswith('camera') else "bz2"
  blob_name =f"{route_name.replace('|', '/')}/{segment_num}/{log_type}.{ext}"
  return get_ci_blob_url(blob_name)


def get_azure_credential():
  if "AZURE_TOKEN" in os.environ:
    return os.environ["AZURE_TOKEN"]
  elif TOKEN_PATH.is_file():
    return TOKEN_PATH.read_text().strip()
  else:
    from azure.identity import AzureCliCredential
    return AzureCliCredential()


def upload_bytes(data: Union[bytes, IO], blob_name: str) -> str:
  from azure.storage.blob import BlobClient
  blob = BlobClient(
    account_url=DATA_CI_ACCOUNT_URL,
    container_name=DATA_CI_CONTAINER,
    blob_name=blob_name,
    credential=get_azure_credential(),
  )
  blob.upload_blob(data)
  return get_ci_blob_url(blob_name)


def upload_file(path: Union[str, os.PathLike], blob_name: str) -> str:
  with open(path, "rb") as f:
    return upload_bytes(f, blob_name)
