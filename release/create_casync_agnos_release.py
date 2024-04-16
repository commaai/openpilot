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



if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="creates a casync release")
  parser.add_argument("output_dir", type=str, help="output directory for the channel")
  parser.add_argument("working_dir", type=str, help="working directory")
  parser.add_argument("--manifest", type=str, help="json manifest to create agnos release from", \
                        default=str(pathlib.Path(BASEDIR) / AGNOS_MANIFEST_FILE))
  args = parser.parse_args()

  output_dir = pathlib.Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  working_dir = pathlib.Path(args.working_dir)
  working_dir.mkdir(parents=True, exist_ok=True)

  manifest_file = pathlib.Path(args.manifest)

  with tempfile.NamedTemporaryFile(dir=str(working_dir)) as entry_file:
    entry_path = pathlib.Path(entry_file.name)

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
      create_casync_from_file(entry_path, output_dir, agnos_filename)
      print(f"created casnc in {time.monotonic() - start}")
