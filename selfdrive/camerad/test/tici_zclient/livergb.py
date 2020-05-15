#!/usr/bin/env python
import numpy as np
import cv2
from time import time, sleep

H, W = (604, 964)

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
    b = dat[::3].reshape(H, W)
    g = dat[1::3].reshape(H, W)
    r = dat[2::3].reshape(H, W)
    rgb = cv2.merge((r, g, b))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow('RGB',rgb)
    cv2.waitKey(20)
    dat.tofile('/tmp/c3rgb.img')