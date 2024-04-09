#!/usr/bin/env python3

import argparse
import os
import pathlib
from openpilot.tools.lib.azure_container import AzureContainer


if __name__ == "__main__":
  del os.environ["AZURE_TOKEN"] # regerenate token for this bucket

  OPENPILOT_RELEASES_CONTAINER = AzureContainer("commadist", "openpilot-releases")

  parser = argparse.ArgumentParser(description='upload casync folder to azure')
  parser.add_argument("casync_dir", type=str, help="casync directory")
  args = parser.parse_args()

  casync_dir = pathlib.Path(args.casync_dir)

  for f in casync_dir.rglob("*"):
    if f.is_file():
      blob_name = f.relative_to(casync_dir)
      print(f"uploading {f} to {blob_name}")
      OPENPILOT_RELEASES_CONTAINER.upload_file(str(f), str(blob_name), overwrite=True)
