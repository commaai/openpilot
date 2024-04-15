#!/usr/bin/env python3
import argparse
import dataclasses
import json
import os
import pathlib

from openpilot.system.hardware.tici.agnos import AGNOS_MANIFEST_FILE, get_partition_path
from openpilot.system.version import get_build_metadata


BASE_URL = "https://commadist.blob.core.windows.net"

OPENPILOT_RELEASES = f"{BASE_URL}/openpilot-releases/openpilot"
AGNOS_RELEASES = f"{BASE_URL}/openpilot-releases/agnos"


def create_partition_manifest(partition):
  agnos_filename = os.path.basename(partition["url"]).split(".")[0]

  return {
    "type": "partition",
    "casync": {
      "caibx": f"{AGNOS_RELEASES}/{agnos_filename}.caibx"
    },
    "path": get_partition_path(0, partition),
    "ab": True,
    "size": partition["size"],
    "full_check": partition["full_check"],
    "hash_raw": partition["hash_raw"],
  }


def create_openpilot_manifest(build_metadata):
  return {
    "type": "path_tarred",
    "path": "/data/openpilot",
    "casync": {
      "caibx": f"{OPENPILOT_RELEASES}/{build_metadata.canonical}.caibx"
    }
  }


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="creates a casync release")
  parser.add_argument("target_dir", type=str, help="directory of the channel to create manifest from")
  parser.add_argument("output_file", type=str, help="output file to put the manifest")
  args = parser.parse_args()

  with open(pathlib.Path(args.target_dir) / AGNOS_MANIFEST_FILE) as f:
    agnos_manifest = json.load(f)

  build_metadata = get_build_metadata(args.target_dir)

  ret = {
    "build_metadata": dataclasses.asdict(build_metadata),
    "manifest": [
      *[create_partition_manifest(entry) for entry in agnos_manifest],
      create_openpilot_manifest(build_metadata)
    ]
  }

  with open(args.output_file, "w") as f:
    f.write(json.dumps(ret, indent=2))
