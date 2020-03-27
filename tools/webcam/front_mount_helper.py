#!/usr/bin/env python
import numpy as np

# copied from common.transformations/camera.py
eon_dcam_focal_length = 860.0 # pixels
webcam_focal_length = 908.0 # pixels

eon_dcam_intrinsics = np.array([
  [eon_dcam_focal_length,   0,   1152/2.],
  [  0,  eon_dcam_focal_length,  864/2.],
  [  0,    0,     1]])

webcam_intrinsics = np.array([
  [webcam_focal_length,   0.,   1280/2.],
  [  0.,  webcam_focal_length,  720/2.],
  [  0.,    0.,     1.]])

cam_id = 2

if __name__ == "__main__":
  import cv2

  trans_webcam_to_eon_front = np.dot(eon_dcam_intrinsics,np.linalg.inv(webcam_intrinsics))

  cap = cv2.VideoCapture(cam_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

  while (True):
    ret, img = cap.read()
    if ret:
      img = cv2.warpPerspective(img, trans_webcam_to_eon_front, (1152,864), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
      img = img[:,-864//2:,:]
      cv2.imshow('preview', img)
      cv2.waitKey(10)