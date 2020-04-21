#!/usr/bin/env python3

# Question: Can a human drive from this data?

import cv2
import numpy as np
import cereal.messaging as messaging
from common.window import Window
from common.transformations.model import MEDMODEL_INPUT_SIZE
from common.transformations.camera import FULL_FRAME_SIZE, eon_intrinsics

from common.transformations.model import get_camera_frame_from_medmodel_frame
from tools.replay.lib.ui_helpers import CalibrationTransformsForWarpMatrix

if __name__ == "__main__":
  sm = messaging.SubMaster(['liveCalibration'])
  frame = messaging.sub_sock('frame', conflate=True)
  win = Window(MEDMODEL_INPUT_SIZE[0], MEDMODEL_INPUT_SIZE[1], double=True)
  calibration = None

  while 1:
    fpkt = messaging.recv_one(frame)
    if len(fpkt.frame.image) == 0:
      continue
    sm.update(timeout=1)
    rgb_img_raw = fpkt.frame.image
    imgff = np.frombuffer(rgb_img_raw, dtype=np.uint8).reshape((FULL_FRAME_SIZE[1], FULL_FRAME_SIZE[0], 3))
    imgff = imgff[:, :, ::-1] # Convert BGR to RGB

    if sm.updated['liveCalibration']:
      intrinsic_matrix = eon_intrinsics
      img_transform = np.array(fpkt.frame.transform).reshape(3,3)
      extrinsic_matrix = np.asarray(sm['liveCalibration'].extrinsicMatrix).reshape(3, 4)
      ke = intrinsic_matrix.dot(extrinsic_matrix)
      warp_matrix = get_camera_frame_from_medmodel_frame(ke)
      calibration = CalibrationTransformsForWarpMatrix(warp_matrix, intrinsic_matrix, extrinsic_matrix)
      transform = np.dot(img_transform, calibration.model_to_full_frame)


    if calibration is not None:
      imgw = cv2.warpAffine(imgff, transform[:2],
        (MEDMODEL_INPUT_SIZE[0], MEDMODEL_INPUT_SIZE[1]),
        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC)

      win.draw(imgw)


