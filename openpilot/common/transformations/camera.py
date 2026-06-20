import itertools
import numpy as np
from dataclasses import dataclass

import openpilot.common.transformations.orientation as orient

## -- hardcoded hardware params --
@dataclass(frozen=True)
class CameraConfig:
  width: int
  height: int
  focal_length: float

  @property
  def size(self):
    return (self.width, self.height)

  @property
  def intrinsics(self):
    # aka 'K' aka camera_frame_from_view_frame
    return np.array([
      [self.focal_length,  0.0, float(self.width)/2],
      [0.0, self.focal_length, float(self.height)/2],
      [0.0,  0.0, 1.0]
    ])

  @property
  def intrinsics_inv(self):
    # aka 'K_inv' aka view_frame_from_camera_frame
    return np.linalg.inv(self.intrinsics)

@dataclass(frozen=True)
class _NoneCameraConfig(CameraConfig):
  width: int = 0
  height: int = 0
  focal_length: float = 0

@dataclass(frozen=True)
class DeviceCameraConfig:
  fcam: CameraConfig
  dcam: CameraConfig
  ecam: CameraConfig

  def all_cams(self):
    for cam in ['fcam', 'dcam', 'ecam']:
      if not isinstance(getattr(self, cam), _NoneCameraConfig):
        yield cam, getattr(self, cam)

_ar_ox_fisheye = CameraConfig(1928, 1208, 567.0)  # focal length probably wrong? magnification is not consistent across frame
_os_fisheye = CameraConfig(2688 // 2, 1520 // 2, 567.0 / 4 * 3)
_ar_ox_config = DeviceCameraConfig(CameraConfig(1928, 1208, 2648.0), _ar_ox_fisheye, _ar_ox_fisheye)
_os_config = DeviceCameraConfig(CameraConfig(2688 // 2, 1520 // 2, 1522.0 * 3 / 4), _os_fisheye, _os_fisheye)
_neo_config = DeviceCameraConfig(CameraConfig(1164, 874, 910.0), CameraConfig(816, 612, 650.0), _NoneCameraConfig())

DEVICE_CAMERAS = {
  # A "device camera" is defined by a device type and sensor

  # sensor type was never set on eon/neo/two
  ("neo", "unknown"): _neo_config,
  # unknown here is AR0231, field was added with OX03C10 support
  ("tici", "unknown"): _ar_ox_config,

  # before deviceState.deviceType was set, assume tici AR config
  ("unknown", "ar0231"): _ar_ox_config,
  ("unknown", "ox03c10"): _ar_ox_config,

  # simulator (emulates a tici)
  ("pc", "unknown"): _ar_ox_config,
}
prods = itertools.product(('tici', 'tizi', 'mici'), (('ar0231', _ar_ox_config), ('ox03c10', _ar_ox_config), ('os04c10', _os_config)))
DEVICE_CAMERAS.update({(d, c[0]): c[1] for d, c in prods})

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


def normalize(img_pts, intrinsics):
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


def denormalize(img_pts, intrinsics, width=np.inf, height=np.inf):
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


def get_calib_from_vp(vp, intrinsics):
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

