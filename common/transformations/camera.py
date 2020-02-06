import numpy as np
import common.transformations.orientation as orient
import math

FULL_FRAME_SIZE = (1164, 874)
W, H = FULL_FRAME_SIZE[0], FULL_FRAME_SIZE[1]
eon_focal_length = FOCAL = 910.0

# aka 'K' aka camera_frame_from_view_frame
eon_intrinsics = np.array([
  [FOCAL,   0.,   W/2.],
  [  0.,  FOCAL,  H/2.],
  [  0.,    0.,     1.]])


leon_dcam_intrinsics = np.array([
  [650,   0,   816//2],
  [  0,  650,  612//2],
  [  0,    0,     1]])

eon_dcam_intrinsics = np.array([
  [860,   0,   1152//2],
  [  0,  860,  864//2],
  [  0,    0,     1]])

# aka 'K_inv' aka view_frame_from_camera_frame
eon_intrinsics_inv = np.linalg.inv(eon_intrinsics)


# device/mesh : x->forward, y-> right, z->down
# view : x->right, y->down, z->forward
device_frame_from_view_frame = np.array([
  [ 0.,  0.,  1.],
  [ 1.,  0.,  0.],
  [ 0.,  1.,  0.]
])
view_frame_from_device_frame = device_frame_from_view_frame.T


def get_calib_from_vp(vp):
  vp_norm = normalize(vp)
  yaw_calib = np.arctan(vp_norm[0])
  pitch_calib = -np.arctan(vp_norm[1]*np.cos(yaw_calib))
  roll_calib = 0
  return roll_calib, pitch_calib, yaw_calib


# aka 'extrinsic_matrix'
# road : x->forward, y -> left, z->up
def get_view_frame_from_road_frame(roll, pitch, yaw, height):
  device_from_road = orient.rot_from_euler([roll, pitch, yaw]).dot(np.diag([1, -1, -1]))
  view_from_road = view_frame_from_device_frame.dot(device_from_road)
  return np.hstack((view_from_road, [[0], [height], [0]]))


def vp_from_ke(m):
  """
  Computes the vanishing point from the product of the intrinsic and extrinsic
  matrices C = KE.

  The vanishing point is defined as lim x->infinity C (x, 0, 0, 1).T
  """
  return (m[0, 0]/m[2,0], m[1,0]/m[2,0])


def vp_from_rpy(rpy):
  e = get_view_frame_from_road_frame(rpy[0], rpy[1], rpy[2], 1.22)
  ke = np.dot(eon_intrinsics, e)
  return vp_from_ke(ke)


def roll_from_ke(m):
  # note: different from calibration.h/RollAnglefromKE: i think that one's just wrong
  return np.arctan2(-(m[1, 0] - m[1, 1] * m[2, 0] / m[2, 1]),
                    -(m[0, 0] - m[0, 1] * m[2, 0] / m[2, 1]))


def normalize(img_pts, intrinsics=eon_intrinsics):
  # normalizes image coordinates
  # accepts single pt or array of pts
  intrinsics_inv = np.linalg.inv(intrinsics)
  img_pts = np.array(img_pts)
  input_shape = img_pts.shape
  img_pts = np.atleast_2d(img_pts)
  img_pts = np.hstack((img_pts, np.ones((img_pts.shape[0],1))))
  img_pts_normalized = img_pts.dot(intrinsics_inv.T)
  img_pts_normalized[(img_pts < 0).any(axis=1)] = np.nan
  return img_pts_normalized[:,:2].reshape(input_shape)


def denormalize(img_pts, intrinsics=eon_intrinsics):
  # denormalizes image coordinates
  # accepts single pt or array of pts
  img_pts = np.array(img_pts)
  input_shape = img_pts.shape
  img_pts = np.atleast_2d(img_pts)
  img_pts = np.hstack((img_pts, np.ones((img_pts.shape[0],1))))
  img_pts_denormalized = img_pts.dot(intrinsics.T)
  img_pts_denormalized[img_pts_denormalized[:,0] > W] = np.nan
  img_pts_denormalized[img_pts_denormalized[:,0] < 0] = np.nan
  img_pts_denormalized[img_pts_denormalized[:,1] > H] = np.nan
  img_pts_denormalized[img_pts_denormalized[:,1] < 0] = np.nan
  return img_pts_denormalized[:,:2].reshape(input_shape)


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
  pt_view[pt_view[:,2] < 0] = np.nan

  pt_img = pt_view/pt_view[:,2:3]
  return pt_img.reshape(input_shape)[:,:2]


def get_camera_frame_from_calib_frame(camera_frame_from_road_frame):
  camera_frame_from_ground = camera_frame_from_road_frame[:, (0, 1, 3)]
  calib_frame_from_ground = np.dot(eon_intrinsics,
                                     get_view_frame_from_road_frame(0, 0, 0, 1.22))[:, (0, 1, 3)]
  ground_from_calib_frame = np.linalg.inv(calib_frame_from_ground)
  camera_frame_from_calib_frame = np.dot(camera_frame_from_ground, ground_from_calib_frame)
  return camera_frame_from_calib_frame


def pretransform_from_calib(calib):
  roll, pitch, yaw, height = calib
  view_frame_from_road_frame = get_view_frame_from_road_frame(roll, pitch, yaw, height)
  camera_frame_from_road_frame = np.dot(eon_intrinsics, view_frame_from_road_frame)
  camera_frame_from_calib_frame = get_camera_frame_from_calib_frame(camera_frame_from_road_frame)
  return np.linalg.inv(camera_frame_from_calib_frame)


def transform_img(base_img,
                 augment_trans=np.array([0,0,0]),
                 augment_eulers=np.array([0,0,0]),
                 from_intr=eon_intrinsics,
                 to_intr=eon_intrinsics,
                 output_size=None,
                 pretransform=None,
                 top_hacks=False,
                 yuv=False,
                 alpha=1.0,
                 beta=0,
                 blur=0):
  import cv2  # pylint: disable=import-error
  cv2.setNumThreads(1)

  if yuv:
    base_img = cv2.cvtColor(base_img, cv2.COLOR_YUV2RGB_I420)

  size = base_img.shape[:2]
  if not output_size:
    output_size = size[::-1]

  cy = from_intr[1,2]
  def get_M(h=1.22):
    quadrangle = np.array([[0, cy + 20],
                           [size[1]-1, cy + 20],
                           [0, size[0]-1],
                           [size[1]-1, size[0]-1]], dtype=np.float32)
    quadrangle_norm = np.hstack((normalize(quadrangle, intrinsics=from_intr), np.ones((4,1))))
    quadrangle_world = np.column_stack((h*quadrangle_norm[:,0]/quadrangle_norm[:,1],
                                        h*np.ones(4),
                                        h/quadrangle_norm[:,1]))
    rot = orient.rot_from_euler(augment_eulers)
    to_extrinsics = np.hstack((rot.T, -augment_trans[:,None]))
    to_KE = to_intr.dot(to_extrinsics)
    warped_quadrangle_full = np.einsum('jk,ik->ij', to_KE, np.hstack((quadrangle_world, np.ones((4,1)))))
    warped_quadrangle = np.column_stack((warped_quadrangle_full[:,0]/warped_quadrangle_full[:,2],
                                         warped_quadrangle_full[:,1]/warped_quadrangle_full[:,2])).astype(np.float32)
    M = cv2.getPerspectiveTransform(quadrangle, warped_quadrangle.astype(np.float32))
    return M

  M = get_M()
  if pretransform is not None:
    M = M.dot(pretransform)
  augmented_rgb = cv2.warpPerspective(base_img, M, output_size, borderMode=cv2.BORDER_REPLICATE)

  if top_hacks:
    cyy = int(math.ceil(to_intr[1,2]))
    M = get_M(1000)
    if pretransform is not None:
      M = M.dot(pretransform)
    augmented_rgb[:cyy] = cv2.warpPerspective(base_img, M, (output_size[0], cyy), borderMode=cv2.BORDER_REPLICATE)

  # brightness and contrast augment
  augmented_rgb = np.clip((float(alpha)*augmented_rgb + beta), 0, 255).astype(np.uint8)

  # gaussian blur
  if blur > 0:
    augmented_rgb = cv2.GaussianBlur(augmented_rgb,(blur*2+1,blur*2+1),cv2.BORDER_DEFAULT)

  if yuv:
    augmented_img = cv2.cvtColor(augmented_rgb, cv2.COLOR_RGB2YUV_I420)
  else:
    augmented_img = augmented_rgb
  return augmented_img


def yuv_crop(frame, output_size, center=None):
  # output_size in camera coordinates so u,v
  # center in array coordinates so row, column
  import cv2  # pylint: disable=import-error
  rgb = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_I420)
  if not center:
    center = (rgb.shape[0]/2, rgb.shape[1]/2)
  rgb_crop = rgb[center[0] - output_size[1]/2: center[0] + output_size[1]/2,
                 center[1] - output_size[0]/2: center[1] + output_size[0]/2]
  return cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2YUV_I420)
