#!/usr/bin/env python3
import os
import urllib.request
import tempfile
import traceback
from urllib.parse import urlparse
from tqdm import tqdm
from openpilot.common.basedir import BASEDIR
from openpilot.tools.lib.route import Route

ROUTES = [
  "140526191c476eaa/00000062--b46a3e1653/1937/2420",
  "b1c51ac59e436919/00000018--223597e5f4/90/453",
  "f73c01590368ee5b/0000000f--12760794cf/573/802",
  "7830b8e854d6713c/0000002a--8e59617918/583/988",
  "98395b7c5b27882e/000000a7--b86b35ab32/94/334",
  "d2c676bbc2b7e5de/00000017--32fb235cb7/146/600",
  "f73c01590368ee5b/00000014--466f1a09da/361/734",
  "f73c01590368ee5b/00000010--2b6e54cfd4/299/863",
  "f73c01590368ee5b/0000000b--91475022ee/371/675",
]
DATA_DIR = os.path.join(BASEDIR, "data", "replay_routes")


def get_available_routes():
  available = []
  for route_name in ROUTES:
    route_base = '/'.join(route_name.split('/')[:2])
    canonical_name = route_base.replace('/', '|')
    base_dir = os.path.join(DATA_DIR, canonical_name)

    if not os.path.exists(base_dir):
      continue

    for seg_dir_name in os.listdir(base_dir):
      seg_dir = os.path.join(base_dir, seg_dir_name)
      if os.path.isdir(seg_dir):
        files = os.listdir(seg_dir)
        if any(f.startswith('rlog.') for f in files) and 'fcamera.hevc' in files and 'ecamera.hevc' in files:
          available.append(canonical_name)
          break

  return available


def main():
  print("downloading replay routes")
  os.makedirs(DATA_DIR, exist_ok=True)

  for route_name in tqdm(ROUTES):
    print(route_name)

    # Parse time range if present (format: route/start_seconds/end_seconds)
    parts = route_name.split('/')
    route_base = '/'.join(parts[:2])
    start_sec = int(parts[2]) if len(parts) > 2 else None
    end_sec = int(parts[3]) if len(parts) > 3 else None

    try:
      route = Route(route_base.replace('/', '|'))
    except Exception:
      print(traceback.format_exc())
      continue

    base_dir = os.path.join(DATA_DIR, route.name.canonical_name)
    for segment in route.segments:
      # Filter by time range: each segment is ~60 seconds
      if start_sec is not None and end_sec is not None:
        seg_start_sec = segment.name.segment_num * 60
        seg_end_sec = (segment.name.segment_num + 1) * 60
        # Include segment if it overlaps with the time range
        if seg_start_sec >= end_sec or seg_end_sec <= start_sec:
          continue

      seg_dir = os.path.join(base_dir, f"{segment.name.time_str}--{segment.name.segment_num}")
      os.makedirs(seg_dir, exist_ok=True)
      print(f'Downloading segment {segment.name.segment_num}')

      for url in [segment.log_path, segment.camera_path, segment.ecamera_path]:
        if url and url.startswith('http'):
          parsed = urlparse(url)
          filename = os.path.basename(parsed.path)
          dest = os.path.join(seg_dir, filename)

          if os.path.exists(dest):
            print(f'Skipping existing file {filename}')
            continue

          try:
            with tempfile.NamedTemporaryFile(dir=seg_dir, delete=False) as tmpfile:
              urllib.request.urlretrieve(url, tmpfile.name)
              os.rename(tmpfile.name, dest)
          except Exception:
            print(traceback.format_exc())


if __name__ == "__main__":
  main()
