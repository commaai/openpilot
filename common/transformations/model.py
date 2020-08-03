import numpy as np

from common.transformations.camera import (FULL_FRAME_SIZE, eon_focal_length,
                                           get_view_frame_from_road_frame,
                                           get_view_frame_from_calib_frame,
                                           vp_from_ke)

# segnet

SEGNET_SIZE = (512, 384)

segnet_frame_from_camera_frame = np.array([
  [float(SEGNET_SIZE[0])/FULL_FRAME_SIZE[0],    0.,          ],
  [     0.,          float(SEGNET_SIZE[1])/FULL_FRAME_SIZE[1]]])


# model

MODEL_INPUT_SIZE = (320, 160)
MODEL_YUV_SIZE = (MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1] * 3 // 2)
MODEL_CX = MODEL_INPUT_SIZE[0]/2.
MODEL_CY = 21.

model_zoom = 1.25
model_height = 1.22

# canonical model transform
model_intrinsics = np.array(
  [[ eon_focal_length / model_zoom,    0. ,  MODEL_CX],
   [   0. ,  eon_focal_length / model_zoom,  MODEL_CY],
   [   0. ,                            0. ,   1.]])


# MED model
MEDMODEL_INPUT_SIZE = (512, 256)
MEDMODEL_YUV_SIZE = (MEDMODEL_INPUT_SIZE[0], MEDMODEL_INPUT_SIZE[1] * 3 // 2)
MEDMODEL_CY = 47.6

medmodel_zoom = 1.
medmodel_intrinsics = np.array(
  [[ eon_focal_length / medmodel_zoom,    0. ,  0.5 * MEDMODEL_INPUT_SIZE[0]],
   [   0. ,  eon_focal_length / medmodel_zoom,  MEDMODEL_CY],
   [   0. ,                            0. ,   1.]])

# CAL model
CALMODEL_INPUT_SIZE = (512, 256)
CALMODEL_YUV_SIZE = (CALMODEL_INPUT_SIZE[0], CALMODEL_INPUT_SIZE[1] * 3 // 2)
CALMODEL_CY = 47.6

calmodel_zoom = 1.5
calmodel_intrinsics = np.array(
  [[ eon_focal_length / calmodel_zoom,    0. ,  0.5 * CALMODEL_INPUT_SIZE[0]],
   [   0. ,  eon_focal_length / calmodel_zoom,  CALMODEL_CY],
   [   0. ,                            0. ,   1.]])


# BIG model

BIGMODEL_INPUT_SIZE = (1024, 512)
BIGMODEL_YUV_SIZE = (BIGMODEL_INPUT_SIZE[0], BIGMODEL_INPUT_SIZE[1] * 3 // 2)

bigmodel_zoom = 1.
bigmodel_intrinsics = np.array(
  [[ eon_focal_length / bigmodel_zoom,    0. , 0.5 * BIGMODEL_INPUT_SIZE[0]],
   [   0. ,  eon_focal_length / bigmodel_zoom,  256+MEDMODEL_CY],
   [   0. ,                            0. ,   1.]])

model_frame_from_road_frame = np.dot(model_intrinsics,
  get_view_frame_from_road_frame(0, 0, 0, model_height))

bigmodel_frame_from_road_frame = np.dot(bigmodel_intrinsics,
  get_view_frame_from_road_frame(0, 0, 0, model_height))

medmodel_frame_from_road_frame = np.dot(medmodel_intrinsics,
  get_view_frame_from_road_frame(0, 0, 0, model_height))

medmodel_frame_from_calib_frame = np.dot(medmodel_intrinsics,
  get_view_frame_from_calib_frame(0, 0, 0, model_height))

model_frame_from_bigmodel_frame = np.dot(model_intrinsics, np.linalg.inv(bigmodel_intrinsics))
medmodel_frame_from_bigmodel_frame = np.dot(medmodel_intrinsics, np.linalg.inv(bigmodel_intrinsics))

# 'camera from model camera'
def get_model_height_transform(camera_frame_from_road_frame, height):
  camera_frame_from_road_ground = np.dot(camera_frame_from_road_frame, np.array([
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 0],
      [0, 0, 1],
  ]))

  camera_frame_from_road_high = np.dot(camera_frame_from_road_frame, np.array([
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, height - model_height],
      [0, 0, 1],
  ]))

  road_high_from_camera_frame = np.linalg.inv(camera_frame_from_road_high)
  high_camera_from_low_camera = np.dot(camera_frame_from_road_ground, road_high_from_camera_frame)

  return high_camera_from_low_camera


# camera_frame_from_model_frame aka 'warp matrix'
# was: calibration.h/CalibrationTransform
def get_camera_frame_from_model_frame(camera_frame_from_road_frame, height=model_height):
  vp = vp_from_ke(camera_frame_from_road_frame)

  model_camera_from_model_frame = np.array([
    [model_zoom,         0., vp[0] - MODEL_CX * model_zoom],
    [        0., model_zoom, vp[1] - MODEL_CY * model_zoom],
    [        0.,         0.,                            1.],
  ])

  # This function is super slow, so skip it if height is very close to canonical
  # TODO: speed it up!
  if abs(height - model_height) > 0.001:
    camera_from_model_camera = get_model_height_transform(camera_frame_from_road_frame, height)
  else:
    camera_from_model_camera = np.eye(3)

  return np.dot(camera_from_model_camera, model_camera_from_model_frame)


def get_camera_frame_from_medmodel_frame(camera_frame_from_road_frame):
  camera_frame_from_ground = camera_frame_from_road_frame[:, (0, 1, 3)]
  medmodel_frame_from_ground = medmodel_frame_from_road_frame[:, (0, 1, 3)]

  ground_from_medmodel_frame = np.linalg.inv(medmodel_frame_from_ground)
  camera_frame_from_medmodel_frame = np.dot(camera_frame_from_ground, ground_from_medmodel_frame)

  return camera_frame_from_medmodel_frame


def get_camera_frame_from_bigmodel_frame(camera_frame_from_road_frame):
  camera_frame_from_ground = camera_frame_from_road_frame[:, (0, 1, 3)]
  bigmodel_frame_from_ground = bigmodel_frame_from_road_frame[:, (0, 1, 3)]

  ground_from_bigmodel_frame = np.linalg.inv(bigmodel_frame_from_ground)
  camera_frame_from_bigmodel_frame = np.dot(camera_frame_from_ground, ground_from_bigmodel_frame)

  return camera_frame_from_bigmodel_frame


def get_model_frame(snu_full, camera_frame_from_model_frame, size):
  idxs = camera_frame_from_model_frame.dot(np.column_stack([np.tile(np.arange(size[0]), size[1]),
                                                            np.tile(np.arange(size[1]), (size[0], 1)).T.flatten(),
                                                            np.ones(size[0] * size[1])]).T).T.astype(int)
  calib_flat = snu_full[idxs[:, 1], idxs[:, 0]]
  if len(snu_full.shape) == 3:
    calib = calib_flat.reshape((size[1], size[0], 3))
  elif len(snu_full.shape) == 2:
    calib = calib_flat.reshape((size[1], size[0]))
  else:
    raise ValueError("shape of input img is weird")
  return calib
