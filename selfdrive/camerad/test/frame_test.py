#!/usr/bin/env python3
import numpy as np
import cereal.messaging as messaging
from PIL import ImageFont, ImageDraw, Image

font = ImageFont.truetype("arial", size=72)
def get_frame(idx):
  img = np.zeros((874, 1164, 3), np.uint8)
  img[100:400, 100:100+(idx % 10) * 100] = 255

  # big number
  im2 = Image.new("RGB", (200, 200))
  draw = ImageDraw.Draw(im2)
  draw.text((10, 100), "%02d" % idx, font=font)
  img[400:600, 400:600] = np.array(im2.getdata()).reshape((200, 200, 3))
  return img.tostring()

if __name__ == "__main__":
  from common.realtime import Ratekeeper
  rk = Ratekeeper(20)

  pm = messaging.PubMaster(['roadCameraState'])
  frm = [get_frame(x) for x in range(30)]
  idx = 0
  while 1:
    print("send %d" % idx)
    dat = messaging.new_message('roadCameraState')
    dat.valid = True
    dat.frame = {
      "frameId": idx,
      "image": frm[idx % len(frm)],
    }
    pm.send('roadCameraState', dat)

    idx += 1
    rk.keep_time()
    #time.sleep(1.0)
