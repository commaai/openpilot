import argparse
import dataclasses
import json
import pathlib

from openpilot.common.run import run_cmd
from openpilot.system.hardware.tici.agnos import AGNOS_MANIFEST_FILE
from openpilot.system.version import get_build_metadata


BASE_URL = "https://commadist.blob.core.windows.net"

CHANNEL_DATA = pathlib.Path(__file__).parent / "channel_data" / "agnos"

OPENPILOT_RELEASES = f"{BASE_URL}/openpilot-releases"
AGNOS_RELEASES = f"{BASE_URL}/agnos-releases"


def create_partition_manifest(agnos_version, partition):
  return {
    "type": "partition",
    "casync": {
      "caibx": f"{AGNOS_RELEASES}/agnos-{agnos_version}-{partition['name']}.caibx"
    },
    "name": partition["name"],
    "size": partition["size"],
    "full_check": partition["full_check"],
    "hash_raw": partition["hash_raw"]
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

  agnos_version = run_cmd(["bash", "-c", r"unset AGNOS_VERSION && source launch_env.sh && \
                          echo -n $AGNOS_VERSION"], args.target_dir).strip()

  build_metadata = get_build_metadata(args.target_dir)

  ret = {
    "build_metadata": dataclasses.asdict(build_metadata),
    "manifest": [
      *[create_partition_manifest(agnos_version, entry) for entry in agnos_manifest],
      create_openpilot_manifest(build_metadata)
    ]
  }

  with open(args.output_file, "w") as f:
    f.write(json.dumps(ret, indent=2))
