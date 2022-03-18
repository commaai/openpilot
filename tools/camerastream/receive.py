#!/usr/bin/env python
import os
import sys
import numpy as np
os.environ['ZMQ'] = '1'

from common.window import Window
import cereal.messaging as messaging

# start camerad with 'SEND_ROAD=1 SEND_DRIVER=1 SEND_WIDE_ROAD=1 XMIN=771 XMAX=1156 YMIN=483 YMAX=724 ./camerad'
# also start bridge
# then run this "./receive.py <ip>"

if "FULL" in os.environ:
  SCALE = 2
  XMIN, XMAX = 0, 1927
  YMIN, YMAX = 0, 1207
else:
  SCALE = 1
  XMIN = 771
  XMAX = 1156
  YMIN = 483
  YMAX = 724
H, W = ((YMAX-YMIN+1)//SCALE, (XMAX-XMIN+1)//SCALE)

if __name__ == '__main__':
  cameras = ['roadCameraState', 'wideRoadCameraState', 'driverCameraState']
  if "CAM" in os.environ:
    cam = int(os.environ['CAM'])
    cameras = cameras[cam:cam+1]
  sm = messaging.SubMaster(cameras, addr=sys.argv[1])
  win = Window(W*len(cameras), H)
  bdat = np.zeros((H, W*len(cameras), 3), dtype=np.uint8)

  while 1:
    sm.update()
    for i,k in enumerate(cameras):
      if sm.updated[k]:
        #print("update", k)
        bgr_raw = sm[k].image
        dat = np.frombuffer(bgr_raw, dtype=np.uint8).reshape(H, W, 3)[:, :, [2,1,0]]
        bdat[:, W*i:W*(i+1)] = dat
    win.draw(bdat)

