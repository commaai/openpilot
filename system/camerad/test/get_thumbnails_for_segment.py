#!/usr/bin/env python3
import argparse
import os
from tqdm import tqdm

from openpilot.tools.lib.logreader import LogReader

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("route", help="The route name")
  args = parser.parse_args()

  out_path = os.path.join("jpegs", f"{args.route.replace('|', '_').replace('/', '_')}")
  os.makedirs(out_path, exist_ok=True)

  lr = LogReader(args.route)

  for msg in tqdm(lr):
    if msg.which() == 'thumbnail':
      with open(os.path.join(out_path, f"{msg.thumbnail.frameId}.jpg"), 'wb') as f:
        f.write(msg.thumbnail.thumbnail)
    elif msg.which() == 'navThumbnail':
      with open(os.path.join(out_path, f"nav_{msg.navThumbnail.frameId}.jpg"), 'wb') as f:
        f.write(msg.navThumbnail.thumbnail)
