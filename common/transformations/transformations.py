import os
import ctypes
import numpy as np

from ctypes import c_double, c_void_p, Structure, POINTER

# Types
class ECEF_C(Structure):
  _fields_ = [("x", c_double), ("y", c_double), ("z", c_double)]

class NED_C(Structure):
  _fields_ = [("n", c_double), ("e", c_double), ("d", c_double)]

class Geodetic_C(Structure):
  _fields_ = [("lat", c_double), ("lon", c_double), ("alt", c_double), ("radians", ctypes.c_bool)]

class Quaternion_C(Structure):
  _fields_ = [("w", c_double), ("x", c_double), ("y", c_double), ("z", c_double)]


# Load Library
def get_libpath():
  # Assuming shared lib is in the same directory
  dir_path = os.path.dirname(os.path.realpath(__file__))
  return os.path.join(dir_path, "libtransformations_c.so")

try:
  _lib = ctypes.CDLL(get_libpath())
except OSError:
  _lib = None  # type: ignore

def _check_lib():
  if _lib is None:
    raise OSError("libtransformations_c.so not found")

# Function signatures
if _lib:
  _lib.geodetic2ecef.argtypes = [Geodetic_C, POINTER(ECEF_C)]
  _lib.ecef2geodetic.argtypes = [ECEF_C, POINTER(Geodetic_C)]

  _lib.euler2quat.argtypes = [POINTER(c_double), POINTER(Quaternion_C)]
  _lib.quat2euler.argtypes = [Quaternion_C, POINTER(c_double)]
  _lib.quat2rot.argtypes = [Quaternion_C, POINTER(c_double)]
  _lib.rot2quat.argtypes = [POINTER(c_double), POINTER(Quaternion_C)]
  _lib.euler2rot.argtypes = [POINTER(c_double), POINTER(c_double)]
  _lib.rot2euler.argtypes = [POINTER(c_double), POINTER(c_double)]
  _lib.rot_matrix.argtypes = [c_double, c_double, c_double, POINTER(c_double)]
  _lib.ecef_euler_from_ned.argtypes = [ECEF_C, POINTER(c_double), POINTER(c_double)]
  _lib.ned_euler_from_ecef.argtypes = [ECEF_C, POINTER(c_double), POINTER(c_double)]

  _lib.localcoord_create.argtypes = [Geodetic_C]
  _lib.localcoord_create.restype = c_void_p
  _lib.localcoord_create_from_ecef.argtypes = [ECEF_C]
  _lib.localcoord_create_from_ecef.restype = c_void_p
  _lib.localcoord_destroy.argtypes = [c_void_p]

  _lib.localcoord_ecef2ned.argtypes = [c_void_p, ECEF_C, POINTER(NED_C)]
  _lib.localcoord_ned2ecef.argtypes = [c_void_p, NED_C, POINTER(ECEF_C)]
  _lib.localcoord_geodetic2ned.argtypes = [c_void_p, Geodetic_C, POINTER(NED_C)]
  _lib.localcoord_ned2geodetic.argtypes = [c_void_p, NED_C, POINTER(Geodetic_C)]

  _lib.localcoord_get_ned2ecef_matrix.argtypes = [c_void_p, POINTER(c_double)]
  _lib.localcoord_get_ecef2ned_matrix.argtypes = [c_void_p, POINTER(c_double)]


# Helpers
def _arr(data):
  return (c_double * len(data))(*data)

def _to_list(c_arr, size):
  return [c_arr[i] for i in range(size)]


def _geo_to_c(g):
  try:
    if len(g) != 3:
      raise ValueError("Geodetic must be size 3")
  except TypeError:
    raise ValueError("Geodetic must be a sequence") from None
  return Geodetic_C(g[0], g[1], g[2], False)

def _ecef_to_c(e):
  try:
    if len(e) != 3:
      raise ValueError("ECEF must be size 3")
  except TypeError:
    raise ValueError("ECEF must be a sequence") from None
  return ECEF_C(e[0], e[1], e[2])

def _ned_to_c(n):
  try:
    if len(n) != 3:
      raise ValueError("NED must be size 3")
  except TypeError:
    raise ValueError("NED must be a sequence") from None
  return NED_C(n[0], n[1], n[2])


# Exposed API
def geodetic2ecef_single(g):
  _check_lib()
  res = ECEF_C()
  _lib.geodetic2ecef(_geo_to_c(g), ctypes.byref(res))
  return [res.x, res.y, res.z]

def ecef2geodetic_single(e):
  _check_lib()
  res = Geodetic_C()
  _lib.ecef2geodetic(_ecef_to_c(e), ctypes.byref(res))
  return [res.lat, res.lon, res.alt]

def euler2quat_single(e):
  _check_lib()
  res = Quaternion_C()
  _lib.euler2quat(_arr(e), ctypes.byref(res))
  return [res.w, res.x, res.y, res.z]

def quat2euler_single(q):
  _check_lib()
  res = (c_double * 3)()
  qc = Quaternion_C(q[0], q[1], q[2], q[3])
  _lib.quat2euler(qc, res)
  return _to_list(res, 3)

def quat2rot_single(q):
  _check_lib()
  res = (c_double * 9)()
  qc = Quaternion_C(q[0], q[1], q[2], q[3])
  _lib.quat2rot(qc, res)
  return np.array(_to_list(res, 9)).reshape((3, 3))

def rot2quat_single(r):
  _check_lib()
  res = Quaternion_C()
  r_flat = np.array(r).flatten()
  _lib.rot2quat(_arr(r_flat), ctypes.byref(res))
  return [res.w, res.x, res.y, res.z]

def euler2rot_single(e):
  _check_lib()
  res = (c_double * 9)()
  _lib.euler2rot(_arr(e), res)
  return np.array(_to_list(res, 9)).reshape((3, 3))

def rot2euler_single(r):
  _check_lib()
  res = (c_double * 3)()
  r_flat = np.array(r).flatten()
  _lib.rot2euler(_arr(r_flat), res)
  return _to_list(res, 3)

def rot_matrix(roll, pitch, yaw):
  _check_lib()
  res = (c_double * 9)()
  _lib.rot_matrix(roll, pitch, yaw, res)
  return np.array(_to_list(res, 9)).reshape((3, 3))

def ecef_euler_from_ned_single(ecef_init, ned_pose):
  _check_lib()
  res = (c_double * 3)()
  _lib.ecef_euler_from_ned(_ecef_to_c(ecef_init), _arr(ned_pose), res)
  return _to_list(res, 3)

def ned_euler_from_ecef_single(ecef_init, ecef_pose):
  _check_lib()
  res = (c_double * 3)()
  _lib.ned_euler_from_ecef(_ecef_to_c(ecef_init), _arr(ecef_pose), res)
  return _to_list(res, 3)


class LocalCoord:
  def __init__(self, geodetic=None, ecef=None):
    _check_lib()
    if geodetic is not None:
      self.lc = _lib.localcoord_create(_geo_to_c(geodetic))
    elif ecef is not None:
      self.lc = _lib.localcoord_create_from_ecef(_ecef_to_c(ecef))
    else:
      raise ValueError("Must provide geodetic or ecef")

  def __del__(self):
    if hasattr(self, 'lc') and self.lc and _lib:
      _lib.localcoord_destroy(self.lc)
      self.lc = None

  @classmethod
  def from_geodetic(cls, geodetic):
    return cls(geodetic=geodetic)

  @classmethod
  def from_ecef(cls, ecef):
    return cls(ecef=ecef)

  def ecef2ned_single(self, ecef):
    res = NED_C()
    _lib.localcoord_ecef2ned(self.lc, _ecef_to_c(ecef), ctypes.byref(res))
    return [res.n, res.e, res.d]

  def ned2ecef_single(self, ned):
    res = ECEF_C()
    _lib.localcoord_ned2ecef(self.lc, _ned_to_c(ned), ctypes.byref(res))
    return [res.x, res.y, res.z]

  def geodetic2ned_single(self, geodetic):
    res = NED_C()
    _lib.localcoord_geodetic2ned(self.lc, _geo_to_c(geodetic), ctypes.byref(res))
    return [res.n, res.e, res.d]

  def ned2geodetic_single(self, ned):
    res = Geodetic_C()
    _lib.localcoord_ned2geodetic(self.lc, _ned_to_c(ned), ctypes.byref(res))
    return [res.lat, res.lon, res.alt]

  @property
  def ned2ecef_matrix(self):
    res = (c_double * 9)()
    _lib.localcoord_get_ned2ecef_matrix(self.lc, res)
    return np.array(_to_list(res, 9)).reshape((3, 3))

  @property
  def ecef2ned_matrix(self):
    res = (c_double * 9)()
    _lib.localcoord_get_ecef2ned_matrix(self.lc, res)
    return np.array(_to_list(res, 9)).reshape((3, 3))

  @property
  def ned_from_ecef_matrix(self):
    return self.ecef2ned_matrix

  @property
  def ecef_from_ned_matrix(self):
    return self.ned2ecef_matrix
