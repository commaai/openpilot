#!/usr/bin/env python3
from openpilot.tools.lib.azure_container import AzureContainer


class OpenpilotCIContainer(AzureContainer):
  CONTAINER = "openpilotci"

class DataCIContainer(AzureContainer):
  CONTAINER = "commadataci"

class OpenpilotPublicDataset(AzureContainer):
  CONTAINER = "openpilotpublicdataset"

class DataProdContainer(AzureContainer):
  ACCOUNT = "commadata2"
  CONTAINER = "commadata2"
