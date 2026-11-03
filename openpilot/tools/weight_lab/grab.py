#!/usr/bin/env python3
"""Save road camera frames as jpg to check what is in a scene.

usage: grab.py ROUTE/SEG FRAME_IDX [FRAME_IDX ...] [--wide]   (frame index within the segment, 20 fps)
Uses the full-res video when it is uploaded, else the qcamera.
"""
import argparse
import os
import subprocess
import urllib.request
from PIL import Image
from openpilot.tools.lib.framereader import FrameReader
from openpilot.tools.lib.route import Route
from openpilot.tools.weight_lab.lab_paths import DATA

OUT = f'{DATA}/frames'


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('seg')
  ap.add_argument('idx', type=int, nargs='+')
  ap.add_argument('--wide', action='store_true')
  args = ap.parse_args()
  route, num = args.seg.rsplit('/', 1)
  s = Route(route).segments[int(num)]
  path = s.ecamera_path if args.wide else s.camera_path
  os.makedirs(OUT, exist_ok=True)
  fr = FrameReader(path) if path else None
  qpath = f"{OUT}/{route.replace('/', '_')}_{num}_qcamera.ts"
  if fr is None and not os.path.exists(qpath):
    urllib.request.urlretrieve(s.qcamera_path, qpath)
  for i in sorted(args.idx):
    p = f"{OUT}/{route.replace('/', '_')}_{num}_{i}{'_w' if args.wide else ''}{'' if fr else '_q'}.jpg"
    if fr is not None:
      img = Image.fromarray(fr.get(i).reshape(fr.h, fr.w, 3))
      img.thumbnail((960, 960))
      img.save(p, quality=80)
    else:
      subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', f'{i / 20:.2f}', '-i', qpath, '-frames:v', '1', p], check=True)
    print(p)


if __name__ == '__main__':
  main()
