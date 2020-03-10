#!/usr/bin/env python
cam_id = 2

if __name__ == "__main__":
  import cv2
  cap = cv2.VideoCapture(cam_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

  while (True):
    ret, img = cap.read()
    if ret:
      # img = img[:,360:,:]
      img = img[:,-360:,:]
      cv2.imshow('preview', img)
      cv2.waitKey(10)