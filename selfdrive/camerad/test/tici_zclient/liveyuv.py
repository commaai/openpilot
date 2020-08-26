# flake8: noqa
# pylint: disable=W

#!/usr/bin/env python
import numpy as np
import cv2
from time import time, sleep

H, W = (256, 512)

if __name__ == '__main__':
  import zmq
  context = zmq.Context()
  socket = context.socket(zmq.PULL)
  socket.bind("tcp://192.168.3.4:7769")
  while True:
    try:
      message = socket.recv()
    except Exception as ex:
      print(ex)
      message = b"123"

    dat = np.frombuffer(message, dtype=np.float32)
    mc = (dat.reshape(H//2, W//2)).astype(np.uint8)

    hist = cv2.calcHist([mc],[0],None,[32],[0,256])
    hist = (H*hist/hist.max()).astype(np.uint8)
    himg = np.zeros((H//2, W//2), dtype=np.uint8)
    for i,b in enumerate(hist):
      himg[H//2-b[0]:,i*(W//2//32):(i+1)*(W//2//32)] = 222

    cv2.imshow('model fov', np.hstack([mc, himg]))
    cv2.waitKey(20)
    dat.tofile('/tmp/c3yuv.img')
