#!/usr/bin/env python3
import sys

if len(sys.argv) < 4:
  print(f"{sys.argv[0]} <route> <segment> <frame number>")
  print('example: ./fetch_image_from_route.py "02c45f73a2e5c6e9|2020-06-01--18-03-08" 3 500')
  exit(0)

import requests
from PIL import Image
from tools.lib.auth_config import get_token
from tools.lib.framereader import FrameReader

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

fr = FrameReader(cameras[segment])
if frame >= fr.frame_count:
  raise Exception("frame %d not found, got %d frames" % (frame, fr.frame_count))

im = Image.fromarray(fr.get(frame, count=1, pix_fmt="rgb24")[0])
fn = "uxxx_"+route.replace("|", "_")+"_%d_%d.png" % (segment, frame)
im.save(fn)
print(f"saved {fn}")

