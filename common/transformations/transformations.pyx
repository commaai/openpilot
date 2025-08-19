# distutils: language = c++
# cython: language_level = 3
from openpilot.common.transformations.transformations cimport Matrix3, Vector3, Quaternion
from openpilot.common.transformations.transformations cimport ECEF, NED, Geodetic

from openpilot.common.transformations.transformations cimport euler2quat as euler2quat_c
from openpilot.common.transformations.transformations cimport quat2euler as quat2euler_c
from openpilot.common.transformations.transformations cimport quat2rot as quat2rot_c
from openpilot.common.transformations.transformations cimport rot2quat as rot2quat_c
from openpilot.common.transformations.transformations cimport euler2rot as euler2rot_c
from openpilot.common.transformations.transformations cimport rot2euler as rot2euler_c
from openpilot.common.transformations.transformations cimport rot_matrix as rot_matrix_c
from openpilot.common.transformations.transformations cimport ecef_euler_from_ned as ecef_euler_from_ned_c
from openpilot.common.transformations.transformations cimport ned_euler_from_ecef as ned_euler_from_ecef_c
from openpilot.common.transformations.transformations cimport geodetic2ecef as geodetic2ecef_c
from openpilot.common.transformations.transformations cimport ecef2geodetic as ecef2geodetic_c
from openpilot.common.transformations.transformations cimport LocalCoord_c


import numpy as np
cimport numpy as np

cdef np.ndarray[double, ndim=2] matrix2numpy(Matrix3 m):
    return np.array([
        [m(0, 0), m(0, 1), m(0, 2)],
        [m(1, 0), m(1, 1), m(1, 2)],
        [m(2, 0), m(2, 1), m(2, 2)],
    ])

cdef Matrix3 numpy2matrix(np.ndarray[double, ndim=2, mode="fortran"] m):
    assert m.shape[0] == 3
    assert m.shape[1] == 3
    return Matrix3(<double*>m.data)

cdef ECEF list2ecef(ecef):
    cdef ECEF e
    e.x = ecef[0]
    e.y = ecef[1]
    e.z = ecef[2]
    return e

cdef NED list2ned(ned):
    cdef NED n
    n.n = ned[0]
    n.e = ned[1]
    n.d = ned[2]
    return n

cdef Geodetic list2geodetic(geodetic):
    cdef Geodetic g
    g.lat = geodetic[0]
    g.lon = geodetic[1]
    g.alt = geodetic[2]
    return g

def euler2quat_single(euler):
    cdef Vector3 e = Vector3(euler[0], euler[1], euler[2])
    cdef Quaternion q = euler2quat_c(e)
    return [q.w(), q.x(), q.y(), q.z()]

def quat2euler_single(quat):
    cdef Quaternion q = Quaternion(quat[0], quat[1], quat[2], quat[3])
    cdef Vector3 e = quat2euler_c(q)
    return [e(0), e(1), e(2)]

def quat2rot_single(quat):
    cdef Quaternion q = Quaternion(quat[0], quat[1], quat[2], quat[3])
    cdef Matrix3 r = quat2rot_c(q)
    return matrix2numpy(r)

def rot2quat_single(rot):
    cdef Matrix3 r = numpy2matrix(np.asfortranarray(rot, dtype=np.double))
    cdef Quaternion q = rot2quat_c(r)
    return [q.w(), q.x(), q.y(), q.z()]

def euler2rot_single(euler):
    cdef Vector3 e = Vector3(euler[0], euler[1], euler[2])
    cdef Matrix3 r = euler2rot_c(e)
    return matrix2numpy(r)

def rot2euler_single(rot):
    cdef Matrix3 r = numpy2matrix(np.asfortranarray(rot, dtype=np.double))
    cdef Vector3 e = rot2euler_c(r)
    return [e(0), e(1), e(2)]

def rot_matrix(roll, pitch, yaw):
    return matrix2numpy(rot_matrix_c(roll, pitch, yaw))

def ecef_euler_from_ned_single(ecef_init, ned_pose):
    cdef ECEF init = list2ecef(ecef_init)
    cdef Vector3 pose = Vector3(ned_pose[0], ned_pose[1], ned_pose[2])

    cdef Vector3 e = ecef_euler_from_ned_c(init, pose)
    return [e(0), e(1), e(2)]

def ned_euler_from_ecef_single(ecef_init, ecef_pose):
    cdef ECEF init = list2ecef(ecef_init)
    cdef Vector3 pose = Vector3(ecef_pose[0], ecef_pose[1], ecef_pose[2])

    cdef Vector3 e = ned_euler_from_ecef_c(init, pose)
    return [e(0), e(1), e(2)]

def geodetic2ecef_single(geodetic):
    cdef Geodetic g = list2geodetic(geodetic)
    cdef ECEF e = geodetic2ecef_c(g)
    return [e.x, e.y, e.z]

def ecef2geodetic_single(ecef):
    cdef ECEF e = list2ecef(ecef)
    cdef Geodetic g = ecef2geodetic_c(e)
    return [g.lat, g.lon, g.alt]


cdef class LocalCoord:
    cdef LocalCoord_c * lc

    def __init__(self, geodetic=None, ecef=None):
        assert (geodetic is not None) or (ecef is not None)
        if geodetic is not None:
            self.lc = new LocalCoord_c(list2geodetic(geodetic))
        elif ecef is not None:
            self.lc = new LocalCoord_c(list2ecef(ecef))

    @property
    def ned2ecef_matrix(self):
        return matrix2numpy(self.lc.ned2ecef_matrix)

    @property
    def ecef2ned_matrix(self):
        return matrix2numpy(self.lc.ecef2ned_matrix)

    @property
    def ned_from_ecef_matrix(self):
        return self.ecef2ned_matrix

    @property
    def ecef_from_ned_matrix(self):
        return self.ned2ecef_matrix

    @classmethod
    def from_geodetic(cls, geodetic):
        return cls(geodetic=geodetic)

    @classmethod
    def from_ecef(cls, ecef):
        return cls(ecef=ecef)

    def ecef2ned_single(self, ecef):
        assert self.lc
        cdef ECEF e = list2ecef(ecef)
        cdef NED n = self.lc.ecef2ned(e)
        return [n.n, n.e, n.d]

    def ned2ecef_single(self, ned):
        assert self.lc
        cdef NED n = list2ned(ned)
        cdef ECEF e = self.lc.ned2ecef(n)
        return [e.x, e.y, e.z]

    def geodetic2ned_single(self, geodetic):
        assert self.lc
        cdef Geodetic g = list2geodetic(geodetic)
        cdef NED n = self.lc.geodetic2ned(g)
        return [n.n, n.e, n.d]

    def ned2geodetic_single(self, ned):
        assert self.lc
        cdef NED n = list2ned(ned)
        cdef Geodetic g = self.lc.ned2geodetic(n)
        return [g.lat, g.lon, g.alt]

    def __dealloc__(self):
        del self.lc
