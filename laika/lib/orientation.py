import numpy as np
from numpy import dot, inner, array, linalg
from .coordinates import LocalCoord


'''
Vectorized functions that transform between
rotation matrices, euler angles and quaternions.
All support lists, array or array of arrays as inputs.
Supports both x2y and y_from_x format (y_from_x preferred!).
'''

def euler2quat(eulers):
  eulers = array(eulers)
  if len(eulers.shape) > 1:
    output_shape = (-1,4)
  else:
    output_shape = (4,)
  eulers = np.atleast_2d(eulers)
  gamma, theta, psi = eulers[:,0],  eulers[:,1],  eulers[:,2]

  cos_half_gamma = np.cos(gamma / 2)
  cos_half_theta = np.cos(theta / 2)
  cos_half_psi = np.cos(psi / 2)
  sin_half_gamma = np.sin(gamma / 2)
  sin_half_theta = np.sin(theta / 2)
  sin_half_psi = np.sin(psi / 2)
  q0 = cos_half_gamma * cos_half_theta * cos_half_psi + sin_half_gamma * sin_half_theta * sin_half_psi
  q1 = sin_half_gamma * cos_half_theta * cos_half_psi - cos_half_gamma * sin_half_theta * sin_half_psi
  q2 = cos_half_gamma * sin_half_theta * cos_half_psi + sin_half_gamma * cos_half_theta * sin_half_psi
  q3 = cos_half_gamma * cos_half_theta * sin_half_psi - sin_half_gamma * sin_half_theta * cos_half_psi

  quats = array([q0, q1, q2, q3]).T
  for i in range(len(quats)):
    if quats[i,0] < 0:
      quats[i] = -quats[i]
  return quats.reshape(output_shape)


def quat2euler(quats):
  quats = array(quats)
  if len(quats.shape) > 1:
    output_shape = (-1,3)
  else:
    output_shape = (3,)
  quats = np.atleast_2d(quats)
  q0, q1, q2, q3 = quats[:,0], quats[:,1], quats[:,2], quats[:,3]

  gamma = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))
  theta = np.arcsin(2 * (q0 * q2 - q3 * q1))
  psi = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))

  eulers = array([gamma, theta, psi]).T
  return eulers.reshape(output_shape)


def quat2rot(quats):
  quats = array(quats)
  input_shape = quats.shape
  quats = np.atleast_2d(quats)
  Rs = np.zeros((quats.shape[0], 3, 3))
  q0 = quats[:, 0]
  q1 = quats[:, 1]
  q2 = quats[:, 2]
  q3 = quats[:, 3]
  Rs[:, 0, 0] = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3
  Rs[:, 0, 1] = 2 * (q1 * q2 - q0 * q3)
  Rs[:, 0, 2] = 2 * (q0 * q2 + q1 * q3)
  Rs[:, 1, 0] = 2 * (q1 * q2 + q0 * q3)
  Rs[:, 1, 1] = q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3
  Rs[:, 1, 2] = 2 * (q2 * q3 - q0 * q1)
  Rs[:, 2, 0] = 2 * (q1 * q3 - q0 * q2)
  Rs[:, 2, 1] = 2 * (q0 * q1 + q2 * q3)
  Rs[:, 2, 2] = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3

  if len(input_shape) < 2:
    return Rs[0]
  return Rs


def rot2quat(rots):
  input_shape = rots.shape
  if len(input_shape) < 3:
    rots = array([rots])
  K3 = np.empty((len(rots), 4, 4))
  K3[:, 0, 0] = (rots[:, 0, 0] - rots[:, 1, 1] - rots[:, 2, 2]) / 3.0
  K3[:, 0, 1] = (rots[:, 1, 0] + rots[:, 0, 1]) / 3.0
  K3[:, 0, 2] = (rots[:, 2, 0] + rots[:, 0, 2]) / 3.0
  K3[:, 0, 3] = (rots[:, 1, 2] - rots[:, 2, 1]) / 3.0
  K3[:, 1, 0] = K3[:, 0, 1]
  K3[:, 1, 1] = (rots[:, 1, 1] - rots[:, 0, 0] - rots[:, 2, 2]) / 3.0
  K3[:, 1, 2] = (rots[:, 2, 1] + rots[:, 1, 2]) / 3.0
  K3[:, 1, 3] = (rots[:, 2, 0] - rots[:, 0, 2]) / 3.0
  K3[:, 2, 0] = K3[:, 0, 2]
  K3[:, 2, 1] = K3[:, 1, 2]
  K3[:, 2, 2] = (rots[:, 2, 2] - rots[:, 0, 0] - rots[:, 1, 1]) / 3.0
  K3[:, 2, 3] = (rots[:, 0, 1] - rots[:, 1, 0]) / 3.0
  K3[:, 3, 0] = K3[:, 0, 3]
  K3[:, 3, 1] = K3[:, 1, 3]
  K3[:, 3, 2] = K3[:, 2, 3]
  K3[:, 3, 3] = (rots[:, 0, 0] + rots[:, 1, 1] + rots[:, 2, 2]) / 3.0
  q = np.empty((len(rots), 4))
  for i in range(len(rots)):
    _, eigvecs = linalg.eigh(K3[i].T)
    eigvecs = eigvecs[:,3:]
    q[i, 0] = eigvecs[-1]
    q[i, 1:] = -eigvecs[:-1].flatten()
    if q[i, 0] < 0:
      q[i] = -q[i]

  if len(input_shape) < 3:
    return q[0]
  return q


def euler2rot(eulers):
  return rotations_from_quats(euler2quat(eulers))


def rot2euler(rots):
  return quat2euler(quats_from_rotations(rots))


quats_from_rotations = rot2quat
quat_from_rot = rot2quat
rotations_from_quats = quat2rot
rot_from_quat= quat2rot
rot_from_quat= quat2rot
euler_from_rot = rot2euler
euler_from_quat = quat2euler
rot_from_euler = euler2rot
quat_from_euler = euler2quat


'''
Random helpers below
'''


def quat_product(q, r):
  t = np.zeros(4)
  t[0] = r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3]
  t[1] = r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2]
  t[2] = r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1]
  t[3] = r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0]
  return t


def rot_matrix(roll, pitch, yaw):
  cr, sr = np.cos(roll), np.sin(roll)
  cp, sp = np.cos(pitch), np.sin(pitch)
  cy, sy = np.cos(yaw), np.sin(yaw)
  rr = array([[1,0,0],[0, cr,-sr],[0, sr, cr]])
  rp = array([[cp,0,sp],[0, 1,0],[-sp, 0, cp]])
  ry = array([[cy,-sy,0],[sy, cy,0],[0, 0, 1]])
  return ry.dot(rp.dot(rr))


def rot(axis, angle):
  # Rotates around an arbitrary axis
  ret_1 = (1 - np.cos(angle)) * array([[axis[0]**2, axis[0] * axis[1], axis[0] * axis[2]], [
    axis[1] * axis[0], axis[1]**2, axis[1] * axis[2]
  ], [axis[2] * axis[0], axis[2] * axis[1], axis[2]**2]])
  ret_2 = np.cos(angle) * np.eye(3)
  ret_3 = np.sin(angle) * array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                              [-axis[1], axis[0], 0]])
  return ret_1 + ret_2 + ret_3


def ecef_euler_from_ned(ned_ecef_init, ned_pose):
  '''
  Got it from here:
  Using Rotations to Build Aerospace Coordinate Systems
  -Don Koks
  '''
  converter = LocalCoord.from_ecef(ned_ecef_init)
  x0 = converter.ned2ecef([1, 0, 0]) - converter.ned2ecef([0, 0, 0])
  y0 = converter.ned2ecef([0, 1, 0]) - converter.ned2ecef([0, 0, 0])
  z0 = converter.ned2ecef([0, 0, 1]) - converter.ned2ecef([0, 0, 0])

  x1 = rot(z0, ned_pose[2]).dot(x0)
  y1 = rot(z0, ned_pose[2]).dot(y0)
  z1 = rot(z0, ned_pose[2]).dot(z0)

  x2 = rot(y1, ned_pose[1]).dot(x1)
  y2 = rot(y1, ned_pose[1]).dot(y1)
  z2 = rot(y1, ned_pose[1]).dot(z1)

  x3 = rot(x2, ned_pose[0]).dot(x2)
  y3 = rot(x2, ned_pose[0]).dot(y2)
  #z3 = rot(x2, ned_pose[0]).dot(z2)

  x0 = array([1, 0, 0])
  y0 = array([0, 1, 0])
  z0 = array([0, 0, 1])

  psi = np.arctan2(inner(x3, y0), inner(x3, x0))
  theta = np.arctan2(-inner(x3, z0), np.sqrt(inner(x3, x0)**2 + inner(x3, y0)**2))
  y2 = rot(z0, psi).dot(y0)
  z2 = rot(y2, theta).dot(z0)
  phi = np.arctan2(inner(y3, z2), inner(y3, y2))

  ret = array([phi, theta, psi])
  return ret


def ned_euler_from_ecef(ned_ecef_init, ecef_poses):
  '''
  Got the math from here:
  Using Rotations to Build Aerospace Coordinate Systems
  -Don Koks

  Also accepts array of ecef_poses and array of ned_ecef_inits.
  Where each row is a pose and an ecef_init.
  '''
  ned_ecef_init = array(ned_ecef_init)
  ecef_poses = array(ecef_poses)
  output_shape = ecef_poses.shape
  ned_ecef_init = np.atleast_2d(ned_ecef_init)
  if ned_ecef_init.shape[0] == 1:
    ned_ecef_init = np.tile(ned_ecef_init[0], (output_shape[0], 1))
  ecef_poses = np.atleast_2d(ecef_poses)

  ned_poses = np.zeros(ecef_poses.shape)
  for i, ecef_pose in enumerate(ecef_poses):
    converter = LocalCoord.from_ecef(ned_ecef_init[i])
    x0 = array([1, 0, 0])
    y0 = array([0, 1, 0])
    z0 = array([0, 0, 1])

    x1 = rot(z0, ecef_pose[2]).dot(x0)
    y1 = rot(z0, ecef_pose[2]).dot(y0)
    z1 = rot(z0, ecef_pose[2]).dot(z0)

    x2 = rot(y1, ecef_pose[1]).dot(x1)
    y2 = rot(y1, ecef_pose[1]).dot(y1)
    z2 = rot(y1, ecef_pose[1]).dot(z1)

    x3 = rot(x2, ecef_pose[0]).dot(x2)
    y3 = rot(x2, ecef_pose[0]).dot(y2)
    #z3 = rot(x2, ecef_pose[0]).dot(z2)

    x0 = converter.ned2ecef([1, 0, 0]) - converter.ned2ecef([0, 0, 0])
    y0 = converter.ned2ecef([0, 1, 0]) - converter.ned2ecef([0, 0, 0])
    z0 = converter.ned2ecef([0, 0, 1]) - converter.ned2ecef([0, 0, 0])

    psi = np.arctan2(inner(x3, y0), inner(x3, x0))
    theta = np.arctan2(-inner(x3, z0), np.sqrt(inner(x3, x0)**2 + inner(x3, y0)**2))
    y2 = rot(z0, psi).dot(y0)
    z2 = rot(y2, theta).dot(z0)
    phi = np.arctan2(inner(y3, z2), inner(y3, y2))
    ned_poses[i] = array([phi, theta, psi])

  return ned_poses.reshape(output_shape)


def ecef2car(car_ecef, psi, theta, points_ecef, ned_converter):
  """
  TODO: add roll rotation
  Converts an array of points in ecef coordinates into
  x-forward, y-left, z-up coordinates
  Parameters
  ----------
  psi: yaw, radian
  theta: pitch, radian
  Returns
  -------
  [x, y, z] coordinates in car frame
  """

  # input is an array of points in ecef cocrdinates
  # output is an array of points in car's coordinate (x-front, y-left, z-up)

  # convert points to NED
  points_ned = []
  for p in points_ecef:
    points_ned.append(ned_converter.ecef2ned_matrix.dot(array(p) - car_ecef))

  points_ned = np.vstack(points_ned).T

  # n, e, d -> x, y, z
  # Calculate relative positions and rotate wrt to heading and pitch of car
  invert_R = array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])

  c, s = np.cos(psi), np.sin(psi)
  yaw_R = array([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]])

  c, s = np.cos(theta), np.sin(theta)
  pitch_R = array([[c, 0., -s], [0., 1., 0.], [s, 0., c]])

  return dot(pitch_R, dot(yaw_R, dot(invert_R, points_ned)))
