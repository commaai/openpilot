import argparse
import json
import pathlib
import tempfile
from openpilot.common.basedir import BASEDIR
from openpilot.system.hardware.tici.agnos import StreamingDecompressor, unsparsify, noop, AGNOS_MANIFEST_FILE
from openpilot.system.updated.casync.common import create_casync_from_file



if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="creates a casync release")
  parser.add_argument("output_dir", type=str, help="output directory for the channel")
  parser.add_argument("version", type=str, help="version of agnos this is")
  parser.add_argument("--manifest", type=str, help="json manifest to create agnos release from", \
                        default=str(pathlib.Path(BASEDIR) / AGNOS_MANIFEST_FILE))
  args = parser.parse_args()

  output_dir = pathlib.Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  manifest_file = pathlib.Path(args.manifest)

  with tempfile.NamedTemporaryFile() as entry_file:
    entry_path = pathlib.Path(entry_file.name)

    with open(manifest_file) as f:
      manifest = json.load(f)

    for entry in manifest:
      print(f"creating casync agnos build from {entry}")
      downloader = StreamingDecompressor(entry['url'])

      parse_func = unsparsify if entry['sparse'] else noop

      parsed_chunks = parse_func(downloader)

      size = entry["size"]

      cur = 0
      with open(entry_path, "wb") as f:
        for chunk in parsed_chunks:
          f.write(chunk)

      create_casync_from_file(entry_path, output_dir, f"agnos-{args.version}-{entry['name']}")
