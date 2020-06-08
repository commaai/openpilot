#!/usr/bin/env python
import numpy as np
import cv2
from time import time, sleep

H, W = (604//3, 964//3)
# H, W = (604, 964)

cam_bufs = np.zeros((3,H,W,3), dtype=np.uint8)

if __name__ == '__main__':
  import zmq
  context = zmq.Context()
  socket = context.socket(zmq.PULL)
  socket.bind("tcp://192.168.2.221:7768")
  while True:
    try:
      message = socket.recv()
    except Exception as ex:
      print(ex)
      message = b"123"

    dat = np.frombuffer(message, dtype=np.uint8)
    cam_id = dat[0]
    # import pdb; pdb.set_trace()
    b = dat[::3].reshape(H, W)
    g = dat[1::3].reshape(H, W)
    r = dat[2::3].reshape(H, W)
    cam_bufs[cam_id] = cv2.merge((r, g, b))
    cam_bufs[cam_id]= cv2.cvtColor(cam_bufs[cam_id], cv2.COLOR_RGB2BGR)
    cv2.imshow('RGB',cam_bufs.reshape((3*H,W,3)))
    cv2.waitKey(20)
    dat.tofile('/tmp/c3rgb.img')