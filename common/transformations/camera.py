import numpy as np

import openpilot.common.transformations.orientation as orient

## -- hardcoded hardware params --
eon_f_focal_length = 910.0
eon_d_focal_length = 650.0
tici_f_focal_length = 2648.0
tici_e_focal_length = tici_d_focal_length = 567.0 # probably wrong? magnification is not consistent across frame

eon_f_frame_size = (1164, 874)
eon_d_frame_size = (816, 612)
tici_f_frame_size = tici_e_frame_size = tici_d_frame_size = (1928, 1208)

# aka 'K' aka camera_frame_from_view_frame
eon_fcam_intrinsics = np.array([
  [eon_f_focal_length,  0.0,  float(eon_f_frame_size[0])/2],
  [0.0,  eon_f_focal_length,  float(eon_f_frame_size[1])/2],
  [0.0,  0.0,                                          1.0]])
eon_intrinsics = eon_fcam_intrinsics # xx

eon_dcam_intrinsics = np.array([
  [eon_d_focal_length,  0.0,  float(eon_d_frame_size[0])/2],
  [0.0,  eon_d_focal_length,  float(eon_d_frame_size[1])/2],
  [0.0,  0.0,                                          1.0]])

tici_fcam_intrinsics = np.array([
  [tici_f_focal_length,  0.0,  float(tici_f_frame_size[0])/2],
  [0.0,  tici_f_focal_length,  float(tici_f_frame_size[1])/2],
  [0.0,  0.0,                                            1.0]])

tici_dcam_intrinsics = np.array([
  [tici_d_focal_length,  0.0,  float(tici_d_frame_size[0])/2],
  [0.0,  tici_d_focal_length,  float(tici_d_frame_size[1])/2],
  [0.0,  0.0,                                            1.0]])

tici_ecam_intrinsics = tici_dcam_intrinsics

# aka 'K_inv' aka view_frame_from_camera_frame
eon_fcam_intrinsics_inv = np.linalg.inv(eon_fcam_intrinsics)
eon_intrinsics_inv = eon_fcam_intrinsics_inv # xx

tici_fcam_intrinsics_inv = np.linalg.inv(tici_fcam_intrinsics)
tici_ecam_intrinsics_inv = np.linalg.inv(tici_ecam_intrinsics)


FULL_FRAME_SIZE = tici_f_frame_size
FOCAL = tici_f_focal_length
fcam_intrinsics = tici_fcam_intrinsics

W, H = FULL_FRAME_SIZE[0], FULL_FRAME_SIZE[1]


# device/mesh : x->forward, y-> right, z->down
# view : x->right, y->down, z->forward
device_frame_from_view_frame = np.array([
  [ 0.,  0.,  1.],
  [ 1.,  0.,  0.],
  [ 0.,  1.,  0.]
])
view_frame_from_device_frame = device_frame_from_view_frame.T


# aka 'extrinsic_matrix'
# road : x->forward, y -> left, z->up
def get_view_frame_from_road_frame(roll, pitch, yaw, height):
  device_from_road = orient.rot_from_euler([roll, pitch, yaw]).dot(np.diag([1, -1, -1]))
  view_from_road = view_frame_from_device_frame.dot(device_from_road)
  return np.hstack((view_from_road, [[0], [height], [0]]))



# aka 'extrinsic_matrix'
def get_view_frame_from_calib_frame(roll, pitch, yaw, height):
  device_from_calib= orient.rot_from_euler([roll, pitch, yaw])
  view_from_calib = view_frame_from_device_frame.dot(device_from_calib)
  return np.hstack((view_from_calib, [[0], [height], [0]]))


def vp_from_ke(m):
  """
  Computes the vanishing point from the product of the intrinsic and extrinsic
  matrices C = KE.

  The vanishing point is defined as lim x->infinity C (x, 0, 0, 1).T
  """
  return (m[0, 0]/m[2, 0], m[1, 0]/m[2, 0])


def roll_from_ke(m):
  # note: different from calibration.h/RollAnglefromKE: i think that one's just wrong
  return np.arctan2(-(m[1, 0] - m[1, 1] * m[2, 0] / m[2, 1]),
                    -(m[0, 0] - m[0, 1] * m[2, 0] / m[2, 1]))


def normalize(img_pts, intrinsics=fcam_intrinsics):
  # normalizes image coordinates
  # accepts single pt or array of pts
  intrinsics_inv = np.linalg.inv(intrinsics)
  img_pts = np.array(img_pts)
  input_shape = img_pts.shape
  img_pts = np.atleast_2d(img_pts)
  img_pts = np.hstack((img_pts, np.ones((img_pts.shape[0], 1))))
  img_pts_normalized = img_pts.dot(intrinsics_inv.T)
  img_pts_normalized[(img_pts < 0).any(axis=1)] = np.nan
  return img_pts_normalized[:, :2].reshape(input_shape)


def denormalize(img_pts, intrinsics=fcam_intrinsics, width=np.inf, height=np.inf):
  # denormalizes image coordinates
  # accepts single pt or array of pts
  img_pts = np.array(img_pts)
  input_shape = img_pts.shape
  img_pts = np.atleast_2d(img_pts)
  img_pts = np.hstack((img_pts, np.ones((img_pts.shape[0], 1), dtype=img_pts.dtype)))
  img_pts_denormalized = img_pts.dot(intrinsics.T)
  if np.isfinite(width):
    img_pts_denormalized[img_pts_denormalized[:, 0] > width] = np.nan
    img_pts_denormalized[img_pts_denormalized[:, 0] < 0] = np.nan
  if np.isfinite(height):
    img_pts_denormalized[img_pts_denormalized[:, 1] > height] = np.nan
    img_pts_denormalized[img_pts_denormalized[:, 1] < 0] = np.nan
  return img_pts_denormalized[:, :2].reshape(input_shape)


def get_calib_from_vp(vp, intrinsics=fcam_intrinsics):
  vp_norm = normalize(vp, intrinsics)
  yaw_calib = np.arctan(vp_norm[0])
  pitch_calib = -np.arctan(vp_norm[1]*np.cos(yaw_calib))
  roll_calib = 0
  return roll_calib, pitch_calib, yaw_calib


def device_from_ecef(pos_ecef, orientation_ecef, pt_ecef):
  # device from ecef frame
  # device frame is x -> forward, y-> right, z -> down
  # accepts single pt or array of pts
  input_shape = pt_ecef.shape
  pt_ecef = np.atleast_2d(pt_ecef)
  ecef_from_device_rot = orient.rotations_from_quats(orientation_ecef)
  device_from_ecef_rot = ecef_from_device_rot.T
  pt_ecef_rel = pt_ecef - pos_ecef
  pt_device = np.einsum('jk,ik->ij', device_from_ecef_rot, pt_ecef_rel)
  return pt_device.reshape(input_shape)


def img_from_device(pt_device):
  # img coordinates from pts in device frame
  # first transforms to view frame, then to img coords
  # accepts single pt or array of pts
  input_shape = pt_device.shape
  pt_device = np.atleast_2d(pt_device)
  pt_view = np.einsum('jk,ik->ij', view_frame_from_device_frame, pt_device)

  # This function should never return negative depths
  pt_view[pt_view[:, 2] < 0] = np.nan

  pt_img = pt_view/pt_view[:, 2:3]
  return pt_img.reshape(input_shape)[:, :2]

