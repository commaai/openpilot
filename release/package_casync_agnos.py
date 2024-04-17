#!/usr/bin/env python3
import argparse
import json
import os
import pathlib
import tempfile
import time
from openpilot.common.basedir import BASEDIR
from openpilot.system.hardware.tici.agnos import StreamingDecompressor, unsparsify, noop, AGNOS_MANIFEST_FILE
from openpilot.system.updated.casync.common import create_casync_from_file
from release.package_casync_build import upload_casync_release



if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="creates a casync release")
  parser.add_argument("--manifest", type=str, help="json manifest to create agnos release from", \
                        default=str(pathlib.Path(BASEDIR) / AGNOS_MANIFEST_FILE))
  args = parser.parse_args()

  manifest_file = pathlib.Path(args.manifest)

  with tempfile.TemporaryDirectory() as temp_dir:
    working_dir = pathlib.Path(temp_dir)
    casync_dir = working_dir / "casync"
    casync_dir.mkdir()

    agnos_casync_dir = casync_dir / "agnos"
    agnos_casync_dir.mkdir()

    entry_path = working_dir / "entry"

    with open(manifest_file) as f:
      manifest = json.load(f)

    for entry in manifest:
      print(f"creating casync agnos build from {entry}")
      start = time.monotonic()
      downloader = StreamingDecompressor(entry['url'])

      parse_func = unsparsify if entry['sparse'] else noop

      parsed_chunks = parse_func(downloader)

      size = entry["size"]

      cur = 0
      with open(entry_path, "wb") as f:
        for chunk in parsed_chunks:
          f.write(chunk)

      print(f"downloaded in {time.monotonic() - start}")

      start = time.monotonic()
      agnos_filename = os.path.basename(entry["url"]).split(".")[0]
      create_casync_from_file(entry_path, agnos_casync_dir, agnos_filename)
      print(f"created casnc in {time.monotonic() - start}")

    upload_casync_release(casync_dir)
