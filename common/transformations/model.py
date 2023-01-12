import numpy as np

from openpilot.common.transformations.camera import (FULL_FRAME_SIZE,
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


### This function mimics the update_calibration logic in modeld.cc
### Manually verified to give similar results to xx.uncommon.utils.transform_img
def get_warp_matrix(rpy_calib, wide_cam=False, big_model=False, tici=True):
  from openpilot.common.transformations.orientation import rot_from_euler
  from openpilot.common.transformations.camera import view_frame_from_device_frame, eon_fcam_intrinsics, tici_ecam_intrinsics, tici_fcam_intrinsics

  if tici and wide_cam:
    intrinsics = tici_ecam_intrinsics
  elif tici:
    intrinsics = tici_fcam_intrinsics
  else:
    intrinsics = eon_fcam_intrinsics

  if big_model:
    sbigmodel_from_calib = sbigmodel_frame_from_calib_frame[:, (0,1,2)]
    calib_from_model = np.linalg.inv(sbigmodel_from_calib)
  else:
    medmodel_from_calib = medmodel_frame_from_calib_frame[:, (0,1,2)]
    calib_from_model = np.linalg.inv(medmodel_from_calib)
  device_from_calib = rot_from_euler(rpy_calib)
  camera_from_calib = intrinsics.dot(view_frame_from_device_frame.dot(device_from_calib))
  warp_matrix = camera_from_calib.dot(calib_from_model)
  return warp_matrix


### This is old, just for debugging
def get_warp_matrix_old(rpy_calib, wide_cam=False, big_model=False, tici=True):
  from openpilot.common.transformations.orientation import rot_from_euler
  from openpilot.common.transformations.camera import view_frame_from_device_frame, eon_fcam_intrinsics, tici_ecam_intrinsics, tici_fcam_intrinsics


  def get_view_frame_from_road_frame(roll, pitch, yaw, height):
    device_from_road = rot_from_euler([roll, pitch, yaw]).dot(np.diag([1, -1, -1]))
    view_from_road = view_frame_from_device_frame.dot(device_from_road)
    return np.hstack((view_from_road, [[0], [height], [0]]))

  if tici and wide_cam:
    intrinsics = tici_ecam_intrinsics
  elif tici:
    intrinsics = tici_fcam_intrinsics
  else:
    intrinsics = eon_fcam_intrinsics

  model_height = 1.22
  if big_model:
    model_from_road = np.dot(sbigmodel_intrinsics,
             get_view_frame_from_road_frame(0, 0, 0, model_height))
  else:
    model_from_road = np.dot(medmodel_intrinsics,
             get_view_frame_from_road_frame(0, 0, 0, model_height))
  ground_from_model = np.linalg.inv(model_from_road[:, (0, 1, 3)])

  E = get_view_frame_from_road_frame(*rpy_calib, 1.22)
  camera_frame_from_road_frame = intrinsics.dot(E)
  camera_frame_from_ground = camera_frame_from_road_frame[:,(0,1,3)]
  warp_matrix = camera_frame_from_ground .dot(ground_from_model)
  return warp_matrix
