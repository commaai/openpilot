import os
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import IO, Union


TOKEN_PATH = Path("/data/azure_token")

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

class AzureContainer:
  def __init__(self, account, container):
    self.ACCOUNT = account
    self.CONTAINER = container

  @property
  def ACCOUNT_URL(self) -> str:
    return f"https://{self.ACCOUNT}.blob.core.windows.net"

  @property
  def BASE_URL(self) -> str:
    return f"{self.ACCOUNT_URL}/{self.CONTAINER}/"

  def get_client_and_key(self):
    from azure.storage.blob import ContainerClient
    client = ContainerClient(self.ACCOUNT_URL, self.CONTAINER, credential=get_azure_credential())
    key = get_container_sas(self.ACCOUNT, self.CONTAINER)
    return client, key

  def get_url(self, route_name: str, segment_num, log_type="rlog") -> str:
    ext = "hevc" if log_type.endswith('camera') else "bz2"
    return self.BASE_URL + f"{route_name.replace('|', '/')}/{segment_num}/{log_type}.{ext}"

  def upload_bytes(self, data: Union[bytes, IO], blob_name: str) -> str:
    from azure.storage.blob import BlobClient
    blob = BlobClient(
      account_url=self.ACCOUNT_URL,
      container_name=self.CONTAINER,
      blob_name=blob_name,
      credential=get_azure_credential(),
      overwrite=False,
    )
    blob.upload_blob(data)
    return self.BASE_URL + blob_name

  def upload_file(self, path: Union[str, os.PathLike], blob_name: str) -> str:
    with open(path, "rb") as f:
      return self.upload_bytes(f, blob_name)
