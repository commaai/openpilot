#!/usr/bin/env python3
import os
import urllib.request
import tempfile
from urllib.parse import urlparse
from tqdm import tqdm
from openpilot.common.basedir import BASEDIR
from openpilot.tools.lib.route import Route

ROUTES = ["98395b7c5b27882e/000000b3--5c18aec824"]
DATA_DIR = os.path.join(BASEDIR, "data", "replay_routes")

print("downloading replay routes")
os.makedirs(DATA_DIR, exist_ok=True)

for route_name in tqdm(ROUTES):
  route = Route(route_name.replace('/', '|'))
  base_dir = os.path.join(DATA_DIR, route.name.canonical_name)

  for segment in route.segments:
    seg_dir = os.path.join(base_dir, f"{segment.name.time_str}--{segment.name.segment_num}")
    os.makedirs(seg_dir, exist_ok=True)

    for url in [segment.log_path, segment.camera_path, segment.ecamera_path]:
      if url and url.startswith('http'):
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)
        dest = os.path.join(seg_dir, filename)

        if os.path.exists(dest):
          print('Skipping', filename)
          continue

        print(f'Downloading {filename}')
        with tempfile.NamedTemporaryFile(dir=seg_dir, delete=False) as tmpfile:
          urllib.request.urlretrieve(url, tmpfile.name)
          os.rename(tmpfile.name, dest)
