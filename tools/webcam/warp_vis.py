#!/usr/bin/env python
import numpy as np

# copied from common.transformations/camera.py
eon_focal_length = 910.0 # pixels
eon_dcam_focal_length = 860.0 # pixels

webcam_focal_length = -908.0/1.5 # pixels

eon_intrinsics = np.array([
  [eon_focal_length,   0.,   1164/2.],
  [  0.,  eon_focal_length,  874/2.],
  [  0.,    0.,     1.]])

eon_dcam_intrinsics = np.array([
  [eon_dcam_focal_length,   0,   1152/2.],
  [  0,  eon_dcam_focal_length,  864/2.],
  [  0,    0,     1]])

webcam_intrinsics = np.array([
  [webcam_focal_length,   0.,   1280/2/1.5],
  [  0.,  webcam_focal_length,  720/2/1.5],
  [  0.,    0.,     1.]])

if __name__ == "__main__":
  import cv2
  trans_webcam_to_eon_rear = np.dot(eon_intrinsics,np.linalg.inv(webcam_intrinsics))
  trans_webcam_to_eon_front = np.dot(eon_dcam_intrinsics,np.linalg.inv(webcam_intrinsics))
  print("trans_webcam_to_eon_rear:\n", trans_webcam_to_eon_rear)
  print("trans_webcam_to_eon_front:\n", trans_webcam_to_eon_front)

  cap = cv2.VideoCapture(1)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 853)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

  while (True):
    ret, img = cap.read()
    if ret:
      # img = cv2.warpPerspective(img, trans_webcam_to_eon_rear, (1164,874), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
      img = cv2.warpPerspective(img, trans_webcam_to_eon_front, (1164,874), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
      print(img.shape, end='\r')
      cv2.imshow('preview', img)
      cv2.waitKey(10)


