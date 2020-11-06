#!/usr/bin/env python3

# Question: Can a human drive from this data?

import os
import cv2  # pylint: disable=import-error
import numpy as np
import cereal.messaging as messaging
from common.window import Window
if os.getenv("BIG") is not None:
  from common.transformations.model import BIGMODEL_INPUT_SIZE as MEDMODEL_INPUT_SIZE
  from common.transformations.model import get_camera_frame_from_bigmodel_frame as get_camera_frame_from_medmodel_frame
else:
  from common.transformations.model import MEDMODEL_INPUT_SIZE
  from common.transformations.model import get_camera_frame_from_medmodel_frame

from tools.replay.lib.ui_helpers import CalibrationTransformsForWarpMatrix, _FULL_FRAME_SIZE, _INTRINSICS

if __name__ == "__main__":
  sm = messaging.SubMaster(['liveCalibration'])
  frame = messaging.sub_sock('frame', conflate=True)
  win = Window(MEDMODEL_INPUT_SIZE[0], MEDMODEL_INPUT_SIZE[1], double=True)
  num_px = 0
  calibration = None
  imgff = None

  while 1:
    fpkt = messaging.recv_one(frame)
    if fpkt is None or len(fpkt.frame.image) == 0:
      continue
    sm.update(timeout=1)
    rgb_img_raw = fpkt.frame.image
    num_px = len(rgb_img_raw) // 3

    if rgb_img_raw and num_px in _FULL_FRAME_SIZE.keys():
      FULL_FRAME_SIZE = _FULL_FRAME_SIZE[num_px]
      imgff = np.frombuffer(rgb_img_raw, dtype=np.uint8).reshape((FULL_FRAME_SIZE[1], FULL_FRAME_SIZE[0], 3))
      imgff = imgff[:, :, ::-1]  # Convert BGR to RGB

    if sm.updated['liveCalibration'] and num_px:
      intrinsic_matrix = _INTRINSICS[num_px]
      img_transform = np.array(fpkt.frame.transform).reshape(3, 3)
      extrinsic_matrix = np.asarray(sm['liveCalibration'].extrinsicMatrix).reshape(3, 4)
      ke = intrinsic_matrix.dot(extrinsic_matrix)
      warp_matrix = get_camera_frame_from_medmodel_frame(ke)
      calibration = CalibrationTransformsForWarpMatrix(num_px, warp_matrix, intrinsic_matrix, extrinsic_matrix)
      transform = np.dot(img_transform, calibration.model_to_full_frame)

    if calibration is not None and imgff is not None:
      imgw = cv2.warpAffine(imgff, transform[:2],
        (MEDMODEL_INPUT_SIZE[0], MEDMODEL_INPUT_SIZE[1]),
        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC)

      win.draw(imgw)
