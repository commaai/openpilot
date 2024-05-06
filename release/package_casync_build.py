#!/usr/bin/env python3

# packages a casync release, uploads to azure, and creates a manifest

import argparse
import dataclasses
import json
import os
import pathlib
import tempfile

from openpilot.system.hardware.tici.agnos import AGNOS_MANIFEST_FILE, get_partition_path
from openpilot.system.updated.casync.common import create_build_metadata_file, create_casync_release
from openpilot.system.version import get_build_metadata
from openpilot.tools.lib.azure_container import AzureContainer


BASE_URL = "https://commadist.blob.core.windows.net"

OPENPILOT_RELEASES = f"{BASE_URL}/openpilot-releases/openpilot"
AGNOS_RELEASES = f"{BASE_URL}/openpilot-releases/agnos"


def create_casync_caibx(target_dir: pathlib.Path, output_dir: pathlib.Path):
  output_dir.mkdir()
  build_metadata = get_build_metadata()
  build_metadata.openpilot.build_style = "release" if os.environ.get("RELEASE", None) is not None else "debug"

  create_build_metadata_file(target_dir, build_metadata)

  digest, caibx = create_casync_release(target_dir, output_dir, build_metadata.canonical)

  print(f"Created casync release from {target_dir} to {caibx} with digest {digest}")


def upload_casync_release(casync_dir: pathlib.Path):
  if "AZURE_TOKEN_OPENPILOT_RELEASES" in os.environ:
    os.environ["AZURE_TOKEN"] = os.environ["AZURE_TOKEN_OPENPILOT_RELEASES"]

  OPENPILOT_RELEASES_CONTAINER = AzureContainer("commadist", "openpilot-releases")

  for f in casync_dir.rglob("*"):
    if f.is_file():
      blob_name = f.relative_to(casync_dir)
      print(f"uploading {f} to {blob_name}")
      OPENPILOT_RELEASES_CONTAINER.upload_file(str(f), str(blob_name), overwrite=True)


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


def create_manifest(target_dir):
  with open(pathlib.Path(target_dir) / AGNOS_MANIFEST_FILE) as f:
    agnos_manifest = json.load(f)

  build_metadata = get_build_metadata(args.target_dir)

  return {
    "build_metadata": dataclasses.asdict(build_metadata),
    "manifest": [
      *[create_partition_manifest(entry) for entry in agnos_manifest],
      create_openpilot_manifest(build_metadata)
    ]
  }



if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="creates a casync release")
  parser.add_argument("target_dir", type=str, help="path to a release build of openpilot to create release from")
  args = parser.parse_args()

  target_dir = pathlib.Path(args.target_dir)

  with tempfile.TemporaryDirectory() as temp_dir:
    casync_dir = pathlib.Path(temp_dir) / "casync"
    casync_dir.mkdir(parents=True)

    manifest_file = pathlib.Path(temp_dir) / "manifest.json"

    create_casync_caibx(target_dir, casync_dir / "openpilot")
    upload_casync_release(casync_dir)
    manifest = create_manifest(target_dir)

    print(json.dumps(manifest, indent=2))
