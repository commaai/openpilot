#!/usr/bin/env python3
import sys
import numpy as np

if len(sys.argv) < 4:
  print("%s <route> <segment> <frame number>" % sys.argv[0])
  print('example: ./fetch_image_from_route.py "4cf7a6ad03080c90|2021-09-29--13-46-36" 3 500')
  exit(0)

import requests
from PIL import Image
from tools.lib.auth_config import get_token
from selfdrive.ui.replay.framereader_pyx import FrameReader # pylint: disable=no-name-in-module, import-error

jwt = get_token()

route = sys.argv[1]
segment = int(sys.argv[2])
frame = int(sys.argv[3])

url = 'https://api.commadotai.com/v1/route/'+sys.argv[1]+"/files"
r = requests.get(url, headers={"Authorization": "JWT "+jwt})
assert r.status_code == 200
print("got api response")

cameras = r.json()['cameras']
if segment >= len(cameras):
  raise Exception("segment %d not found, got %d segments" % (segment, len(cameras)))

fr = FrameReader()
fr.load(cameras[segment])
if frame >= fr.frame_count:
  raise Exception("frame %d not found, got %d frames" % (frame, fr.frame_count))

rgb = fr.get(frame)
img = np.frombuffer(rgb, dtype=np.uint8).reshape((fr.height, fr.width, 3))
img = img[:, :, ::-1]
im = Image.fromarray(img)
fn = "uxxx_"+route.replace("|", "_")+"_%d_%d.png" % (segment, frame)
im.save(fn)
print("saved %s" % fn)

