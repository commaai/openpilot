#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cairosvg",
# ]
# ///

import os
import glob
import argparse
from pathlib import Path
import cairosvg

SVG_SCALE = 4

def rasterize_svgs(directory=None, force=False):
  if directory is None:
    directory = os.path.dirname(os.path.abspath(__file__))

  svg_files = glob.glob(os.path.join(directory, "*.svg"))
  print(f"Found {len(svg_files)} SVG files to convert")

  converted = 0
  skipped = 0

  for svg_file in svg_files:
    svg_path = Path(svg_file)
    png_path = svg_path.with_suffix(".png")

    # Check if PNG exists and is newer than SVG
    if not force and png_path.exists():
      svg_mtime = svg_path.stat().st_mtime
      png_mtime = png_path.stat().st_mtime

      if png_mtime >= svg_mtime:
        print(f"Skipping {svg_path.name} (PNG is up to date)")
        skipped += 1
        continue

    print(f"Converting {svg_path.name} to {png_path.name}")
    try:
      cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), scale=SVG_SCALE)
      converted += 1
    except Exception as e:
      print(f"Failed to convert {svg_path.name}: {e}")

  return converted, skipped

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Rasterize SVG files to PNG")
  parser.add_argument("directory", nargs="?", help="Directory containing SVG files (default: script location)")
  parser.add_argument("-f", "--force", action="store_true", help="Force regeneration of all PNGs")
  args = parser.parse_args()

  converted, skipped = rasterize_svgs(args.directory, args.force)
  print(f"Finished: {converted} converted, {skipped} skipped")
