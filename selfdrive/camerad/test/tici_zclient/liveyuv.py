#!/usr/bin/env python
import numpy as np
import cv2
from time import time, sleep

H, W = (256, 512)

if __name__ == '__main__':
  import zmq
  context = zmq.Context()
  socket = context.socket(zmq.PULL)
  socket.bind("tcp://192.168.2.221:7769")
  while True:
    try:
      message = socket.recv()
    except Exception as ex:
      print(ex)
      message = b"123"

    dat = np.frombuffer(message, dtype=np.float32)
    mc = (dat.reshape(H//2, W//2) * 128 + 128).astype(np.uint8)
    cv2.imshow('model fov', mc)
    cv2.waitKey(20)
    dat.tofile('/home/batman/c3yuv.img')