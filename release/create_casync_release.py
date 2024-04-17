#!/usr/bin/env python

import argparse
import os
import pathlib

from openpilot.system.updated.casync.common import create_casync_release, create_build_metadata_file
from openpilot.system.version import get_build_metadata


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="creates a casync release")
  parser.add_argument("target_dir", type=str, help="target directory to build channel from")
  parser.add_argument("output_dir", type=str, help="output directory for the channel")
  args = parser.parse_args()

  target_dir = pathlib.Path(args.target_dir)
  output_dir = pathlib.Path(args.output_dir)

  build_metadata = get_build_metadata()
  build_metadata.openpilot.build_style = "release" if os.environ.get("RELEASE", None) is not None else "debug"

  create_build_metadata_file(target_dir, build_metadata)

  digest, caibx = create_casync_release(target_dir, output_dir, build_metadata.canonical)

  print(f"Created casync release from {target_dir} to {caibx} with digest {digest}")
