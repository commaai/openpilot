import numpy as np


# Constants
a = 6378137.0
b = 6356752.3142
esq = 6.69437999014e-3
e1sq = 6.73949674228e-3


def geodetic2ecef_single(g):
  """
  Convert geodetic coordinates (latitude, longitude, altitude) to ECEF.
  """
  try:
    if len(g) != 3:
      raise ValueError("Geodetic must be size 3")
  except TypeError:
    raise ValueError("Geodetic must be a sequence of length 3") from None

  lat, lon, alt = g
  lat = np.radians(lat)
  lon = np.radians(lon)
  xi = np.sqrt(1.0 - esq * np.sin(lat)**2)
  x = (a / xi + alt) * np.cos(lat) * np.cos(lon)
  y = (a / xi + alt) * np.cos(lat) * np.sin(lon)
  z = (a / xi * (1.0 - esq) + alt) * np.sin(lat)
  return np.array([x, y, z])


def ecef2geodetic_single(e):
  """
  Convert ECEF to geodetic coordinates using Ferrari's solution.
  """
  x, y, z = e
  r = np.sqrt(x**2 + y**2)
  Esq = a**2 - b**2
  F = 54 * b**2 * z**2
  G = r**2 + (1 - esq) * z**2 - esq * Esq
  C = (esq**2 * F * r**2) / (G**3)
  S = np.cbrt(1 + C + np.sqrt(C**2 + 2 * C))
  P = F / (3 * (S + 1 / S + 1)**2 * G**2)
  Q = np.sqrt(1 + 2 * esq**2 * P)
  r_0 = -(P * esq * r) / (1 + Q) + np.sqrt(0.5 * a**2 * (1 + 1.0 / Q) - P * (1 - esq) * z**2 / (Q * (1 + Q)) - 0.5 * P * r**2)
  U = np.sqrt((r - esq * r_0)**2 + z**2)
  V = np.sqrt((r - esq * r_0)**2 + (1 - esq) * z**2)
  Z_0 = b**2 * z / (a * V)
  h = U * (1 - b**2 / (a * V))
  lat = np.arctan((z + e1sq * Z_0) / r)
  lon = np.arctan2(y, x)
  return np.array([np.degrees(lat), np.degrees(lon), h])


def euler2quat_single(euler):
  """
  Convert Euler angles (roll, pitch, yaw) to a quaternion.
  Rotation order: Z-Y-X (yaw, pitch, roll).
  """
  phi, theta, psi = euler

  c_phi, s_phi = np.cos(phi / 2), np.sin(phi / 2)
  c_theta, s_theta = np.cos(theta / 2), np.sin(theta / 2)
  c_psi, s_psi = np.cos(psi / 2), np.sin(psi / 2)

  w = c_phi * c_theta * c_psi + s_phi * s_theta * s_psi
  x = s_phi * c_theta * c_psi - c_phi * s_theta * s_psi
  y = c_phi * s_theta * c_psi + s_phi * c_theta * s_psi
  z = c_phi * c_theta * s_psi - s_phi * s_theta * c_psi

  if w < 0:
    return np.array([-w, -x, -y, -z])
  return np.array([w, x, y, z])


def quat2euler_single(q):
  """
  Convert a quaternion to Euler angles (roll, pitch, yaw).
  """
  w, x, y, z = q
  gamma = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
  sin_arg = 2 * (w * y - z * x)
  sin_arg = np.clip(sin_arg, -1.0, 1.0)
  theta = np.arcsin(sin_arg)
  psi = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
  return np.array([gamma, theta, psi])


def quat2rot_single(q):
  """
  Convert a quaternion to a 3x3 rotation matrix.
  """
  w, x, y, z = q
  xx, yy, zz = x * x, y * y, z * z
  xy, xz, yz = x * y, x * z, y * z
  wx, wy, wz = w * x, w * y, w * z

  mat = np.array([
    [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
    [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
    [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
  ])
  return mat


def rot2quat_single(rot):
  """
  Convert a 3x3 rotation matrix to a quaternion.
  """
  trace = np.trace(rot)
  if trace > 0:
    s = 0.5 / np.sqrt(trace + 1.0)
    w = 0.25 / s
    x = (rot[2, 1] - rot[1, 2]) * s
    y = (rot[0, 2] - rot[2, 0]) * s
    z = (rot[1, 0] - rot[0, 1]) * s
  else:
    if rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
      s = 2.0 * np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
      w = (rot[2, 1] - rot[1, 2]) / s
      x = 0.25 * s
      y = (rot[0, 1] + rot[1, 0]) / s
      z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
      s = 2.0 * np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
      w = (rot[0, 2] - rot[2, 0]) / s
      x = (rot[0, 1] + rot[1, 0]) / s
      y = 0.25 * s
      z = (rot[1, 2] + rot[2, 1]) / s
    else:
      s = 2.0 * np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
      w = (rot[1, 0] - rot[0, 1]) / s
      x = (rot[0, 2] + rot[2, 0]) / s
      y = (rot[1, 2] + rot[2, 1]) / s
      z = 0.25 * s

  if w < 0:
    return np.array([-w, -x, -y, -z])
  return np.array([w, x, y, z])


def euler2rot_single(euler):
  """
  Convert Euler angles (roll, pitch, yaw) to a 3x3 rotation matrix.
  Rotation order: Z-Y-X (yaw, pitch, roll).
  """
  phi, theta, psi = euler

  cx, sx = np.cos(phi), np.sin(phi)
  cy, sy = np.cos(theta), np.sin(theta)
  cz, sz = np.cos(psi), np.sin(psi)

  Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
  Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
  Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

  return Rz @ Ry @ Rx


def rot2euler_single(rot):
  """
  Convert a 3x3 rotation matrix to Euler angles (roll, pitch, yaw).
  """
  return quat2euler_single(rot2quat_single(rot))


def rot_matrix(roll, pitch, yaw):
  """
  Create a 3x3 rotation matrix from roll, pitch, and yaw angles.
  """
  return euler2rot_single([roll, pitch, yaw])


def axis_angle_to_rot(axis, angle):
  """
  Convert an axis-angle representation to a 3x3 rotation matrix.
  """
  c = np.cos(angle / 2)
  s = np.sin(angle / 2)
  q = np.array([c, s*axis[0], s*axis[1], s*axis[2]])
  return quat2rot_single(q)


class LocalCoord:
  """
  A class to handle conversions between ECEF and local NED coordinates.
  """
  def __init__(self, geodetic=None, ecef=None):
    """
    Initialize LocalCoord with either geodetic or ECEF coordinates.
    """
    if geodetic is not None:
      self.init_ecef = geodetic2ecef_single(geodetic)
      lat, lon, _ = geodetic
    elif ecef is not None:
      self.init_ecef = np.array(ecef)
      lat, lon, _ = ecef2geodetic_single(ecef)
    else:
      raise ValueError("Must provide geodetic or ecef")

    lat = np.radians(lat)
    lon = np.radians(lon)

    self.ned2ecef_matrix = np.array([
      [-np.sin(lat) * np.cos(lon), -np.sin(lon), -np.cos(lat) * np.cos(lon)],
      [-np.sin(lat) * np.sin(lon), np.cos(lon), -np.cos(lat) * np.sin(lon)],
      [np.cos(lat), 0, -np.sin(lat)]
    ])
    self.ecef2ned_matrix = self.ned2ecef_matrix.T

  @classmethod
  def from_geodetic(cls, geodetic):
    """
    Create a LocalCoord instance from geodetic coordinates.
    """
    return cls(geodetic=geodetic)

  @classmethod
  def from_ecef(cls, ecef):
    """
    Create a LocalCoord instance from ECEF coordinates.
    """
    return cls(ecef=ecef)

  def ecef2ned_single(self, ecef):
    """
    Convert a single ECEF point to NED coordinates relative to the origin.
    """
    return self.ecef2ned_matrix @ (ecef - self.init_ecef)

  def ned2ecef_single(self, ned):
    """
    Convert a single NED point to ECEF coordinates.
    """
    return self.ned2ecef_matrix @ ned + self.init_ecef

  def geodetic2ned_single(self, geodetic):
    """
    Convert a single geodetic point to NED coordinates.
    """
    ecef = geodetic2ecef_single(geodetic)
    return self.ecef2ned_single(ecef)

  def ned2geodetic_single(self, ned):
    """
    Convert a single NED point to geodetic coordinates.
    """
    ecef = self.ned2ecef_single(ned)
    return ecef2geodetic_single(ecef)

  @property
  def ned_from_ecef_matrix(self):
    """
    Returns the rotation matrix from ECEF to NED coordinates.
    """
    return self.ecef2ned_matrix

  @property
  def ecef_from_ned_matrix(self):
    """
    Returns the rotation matrix from NED to ECEF coordinates.
    """
    return self.ned2ecef_matrix


def ecef_euler_from_ned_single(ecef_init, ned_pose):
  """
  Convert NED Euler angles (roll, pitch, yaw) at a given ECEF origin
  to equivalent ECEF Euler angles.
  """
  converter = LocalCoord(ecef=ecef_init)
  zero = np.array(ecef_init)

  x0 = converter.ned2ecef_single([1, 0, 0]) - zero
  y0 = converter.ned2ecef_single([0, 1, 0]) - zero
  z0 = converter.ned2ecef_single([0, 0, 1]) - zero

  phi, theta, psi = ned_pose

  x1 = axis_angle_to_rot(z0, psi) @ x0
  y1 = axis_angle_to_rot(z0, psi) @ y0
  z1 = axis_angle_to_rot(z0, psi) @ z0

  x2 = axis_angle_to_rot(y1, theta) @ x1
  y2 = axis_angle_to_rot(y1, theta) @ y1
  z2 = axis_angle_to_rot(y1, theta) @ z1

  x3 = axis_angle_to_rot(x2, phi) @ x2
  y3 = axis_angle_to_rot(x2, phi) @ y2

  x0 = np.array([1.0, 0, 0])
  y0 = np.array([0, 1.0, 0])
  z0 = np.array([0, 0, 1.0])

  psi_out = np.arctan2(np.dot(x3, y0), np.dot(x3, x0))
  theta_out = np.arctan2(-np.dot(x3, z0), np.sqrt(np.dot(x3, x0)**2 + np.dot(x3, y0)**2))

  y2 = axis_angle_to_rot(z0, psi_out) @ y0
  z2 = axis_angle_to_rot(y2, theta_out) @ z0

  phi_out = np.arctan2(np.dot(y3, z2), np.dot(y3, y2))

  return np.array([phi_out, theta_out, psi_out])


def ned_euler_from_ecef_single(ecef_init, ecef_pose):
  """
  Convert ECEF Euler angles (roll, pitch, yaw) at a given ECEF origin
  to equivalent NED Euler angles.
  """
  converter = LocalCoord(ecef=ecef_init)

  x0 = np.array([1.0, 0, 0])
  y0 = np.array([0, 1.0, 0])
  z0 = np.array([0, 0, 1.0])

  phi, theta, psi = ecef_pose

  x1 = axis_angle_to_rot(z0, psi) @ x0
  y1 = axis_angle_to_rot(z0, psi) @ y0
  z1 = axis_angle_to_rot(z0, psi) @ z0

  x2 = axis_angle_to_rot(y1, theta) @ x1
  y2 = axis_angle_to_rot(y1, theta) @ y1
  z2 = axis_angle_to_rot(y1, theta) @ z1

  x3 = axis_angle_to_rot(x2, phi) @ x2
  y3 = axis_angle_to_rot(x2, phi) @ y2

  zero = np.array(ecef_init)
  x0 = converter.ned2ecef_single([1, 0, 0]) - zero
  y0 = converter.ned2ecef_single([0, 1, 0]) - zero
  z0 = converter.ned2ecef_single([0, 0, 1]) - zero

  psi_out = np.arctan2(np.dot(x3, y0), np.dot(x3, x0))
  theta_out = np.arctan2(-np.dot(x3, z0), np.sqrt(np.dot(x3, x0)**2 + np.dot(x3, y0)**2))

  y2 = axis_angle_to_rot(z0, psi_out) @ y0
  z2 = axis_angle_to_rot(y2, theta_out) @ z0

  phi_out = np.arctan2(np.dot(y3, z2), np.dot(y3, y2))

  return np.array([phi_out, theta_out, psi_out])
