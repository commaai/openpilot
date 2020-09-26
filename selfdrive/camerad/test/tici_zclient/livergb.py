# flake8: noqa
# pylint: disable=W

#!/usr/bin/env python
import numpy as np
import cv2
from time import time, sleep

H, W = (604*2//6, 964*2//6)
# H, W = (604, 964)

cam_bufs = np.zeros((3,H,W,3), dtype=np.uint8)
hist_bufs = np.zeros((3,H,200,3), dtype=np.uint8)

if __name__ == '__main__':
  import zmq
  context = zmq.Context()
  socket = context.socket(zmq.PULL)
  socket.bind("tcp://192.168.3.4:7768")
  while True:
    try:
      message = socket.recv()
    except Exception as ex:
      print(ex)
      message = b"123"

    dat = np.frombuffer(message, dtype=np.uint8)
    cam_id = (dat[0] + 1) % 3
    # import pdb; pdb.set_trace()
    b = dat[::3].reshape(H, W)
    g = dat[1::3].reshape(H, W)
    r = dat[2::3].reshape(H, W)
    cam_bufs[cam_id] = cv2.merge((r, g, b))
    cam_bufs[cam_id]= cv2.cvtColor(cam_bufs[cam_id], cv2.COLOR_RGB2BGR)

    hist = cv2.calcHist([cv2.cvtColor(cam_bufs[cam_id], cv2.COLOR_BGR2GRAY)],[0],None,[32],[0,256])
    hist = (H*hist/hist.max()).astype(np.uint8)
    for i,bb in enumerate(hist):
        hist_bufs[cam_id, H-bb[0]:,i*(200//32):(i+1)*(200//32), :] = (222,222,222)

    out = cam_bufs.reshape((3*H,W,3))
    hist_bufs_out = hist_bufs.reshape((3*H,200,3))
    out = np.hstack([out, hist_bufs_out])
    cv2.imshow('RGB', out)
    cv2.waitKey(55)
    #dat.tofile('/tmp/c3rgb.img')
    #cv2.imwrite('/tmp/c3rgb.png', out)
