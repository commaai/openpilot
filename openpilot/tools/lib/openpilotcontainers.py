#!/usr/bin/env python3
from openpilot.tools.lib.azure_container import AzureContainer

OpenpilotCIContainer = AzureContainer("commadataci", "openpilotci")
DataCIContainer = AzureContainer("commadataci", "commadataci")
DataProdContainer = AzureContainer("commadata2", "commadata2")
