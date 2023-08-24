#!/usr/bin/env python3

import os

from tqdm import tqdm

from openpilot.common.file_helpers import mkdirs_exists_ok
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.route import Route

import argparse

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("route", help="The route name")
  parser.add_argument("segment", type=int,  help="The index of the segment")
  args = parser.parse_args()

  out_path = os.path.join("jpegs", f"{args.route.replace('|', '_')}_{args.segment}")
  mkdirs_exists_ok(out_path)

  r = Route(args.route)
  path = r.log_paths()[args.segment] or r.qlog_paths()[args.segment]
  lr = list(LogReader(path))

  for msg in tqdm(lr):
    if msg.which() == 'thumbnail':
      with open(os.path.join(out_path, f"{msg.thumbnail.frameId}.jpg"), 'wb') as f:
        f.write(msg.thumbnail.thumbnail)
    elif msg.which() == 'navThumbnail':
      with open(os.path.join(out_path, f"nav_{msg.navThumbnail.frameId}.jpg"), 'wb') as f:
        f.write(msg.navThumbnail.thumbnail)
