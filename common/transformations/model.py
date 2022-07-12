import numpy as np

from common.transformations.camera import (FULL_FRAME_SIZE,
                                           get_view_frame_from_calib_frame)

# segnet
SEGNET_SIZE = (512, 384)

def get_segnet_frame_from_camera_frame(segnet_size=SEGNET_SIZE, full_frame_size=FULL_FRAME_SIZE):
  return np.array([[float(segnet_size[0]) / full_frame_size[0],  0.0],
                   [0.0,  float(segnet_size[1]) / full_frame_size[1]]])
segnet_frame_from_camera_frame = get_segnet_frame_from_camera_frame() # xx


# MED model
MEDMODEL_INPUT_SIZE = (512, 256)
MEDMODEL_YUV_SIZE = (MEDMODEL_INPUT_SIZE[0], MEDMODEL_INPUT_SIZE[1] * 3 // 2)
MEDMODEL_CY = 47.6

medmodel_fl = 910.0
medmodel_intrinsics = np.array([
  [medmodel_fl,  0.0,  0.5 * MEDMODEL_INPUT_SIZE[0]],
  [0.0,  medmodel_fl,                   MEDMODEL_CY],
  [0.0,  0.0,                                   1.0]])


# BIG model
BIGMODEL_INPUT_SIZE = (1024, 512)
BIGMODEL_YUV_SIZE = (BIGMODEL_INPUT_SIZE[0], BIGMODEL_INPUT_SIZE[1] * 3 // 2)

bigmodel_fl = 910.0
bigmodel_intrinsics = np.array([
  [bigmodel_fl,  0.0,  0.5 * BIGMODEL_INPUT_SIZE[0]],
  [0.0,  bigmodel_fl,             256 + MEDMODEL_CY],
  [0.0,  0.0,                                   1.0]])


# SBIG model (big model with the size of small model)
SBIGMODEL_INPUT_SIZE = (512, 256)
SBIGMODEL_YUV_SIZE = (SBIGMODEL_INPUT_SIZE[0], SBIGMODEL_INPUT_SIZE[1] * 3 // 2)

sbigmodel_fl = 455.0
sbigmodel_intrinsics = np.array([
  [sbigmodel_fl,  0.0,  0.5 * SBIGMODEL_INPUT_SIZE[0]],
  [0.0,  sbigmodel_fl,      0.5 * (256 + MEDMODEL_CY)],
  [0.0,  0.0,                                     1.0]])

bigmodel_frame_from_calib_frame = np.dot(bigmodel_intrinsics,
  get_view_frame_from_calib_frame(0, 0, 0, 0))


sbigmodel_frame_from_calib_frame = np.dot(sbigmodel_intrinsics,
  get_view_frame_from_calib_frame(0, 0, 0, 0))

medmodel_frame_from_calib_frame = np.dot(medmodel_intrinsics,
  get_view_frame_from_calib_frame(0, 0, 0, 0))

medmodel_frame_from_bigmodel_frame = np.dot(medmodel_intrinsics, np.linalg.inv(bigmodel_intrinsics))
