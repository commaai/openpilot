#!/usr/bin/env python3
# This file does the same thing as webcam.cc, but in python...
# It is now obsolete but it is useful for prototyping.
from common.realtime import Ratekeeper
import cereal.messaging as messaging
import numpy as np
import threading
import cv2

FRAME_WIDTH, FRAME_HEIGHT = 1164, 874

def health_function():
  pm = messaging.PubMaster(['health'])
  rk = Ratekeeper(1.0)
  while (True):
    dat = messaging.new_message()
    dat.init('health')
    dat.valid = True
    dat.health = {
      'ignitionLine': True,
      'hwType': "whitePanda",
      'controlsAllowed': True
    }
    pm.send('health', dat)
    rk.keep_time()

def frame_function():
  cap = cv2.VideoCapture(0)
  pm = messaging.PubMaster(['frame'])

  while (True):
    ret, img = cap.read()
    
    if ret:
      img = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))

      dat = messaging.new_message()
      dat.init('frame')
      dat.frame = {
        #"frameId": frame_id,
        "image": img.tostring(),
      }
      pm.send('frame', dat)

if __name__=="__main__":
  threading.Thread(target=health_function).start()
  frame_function()

  cap.release()
  cv2.destroyAllWindows()
