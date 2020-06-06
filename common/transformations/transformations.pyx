from transformations cimport Matrix3, Vector3, Quaternion
from transformations cimport ECEF, NED, Geodetic

from transformations cimport euler2quat as euler2quat_c
from transformations cimport quat2euler as quat2euler_c
from transformations cimport quat2rot as quat2rot_c
from transformations cimport rot2quat as rot2quat_c
from transformations cimport euler2rot as euler2rot_c
from transformations cimport rot2euler as rot2euler_c
from transformations cimport rot_matrix as rot_matrix_c
from transformations cimport ecef_euler_from_ned as ecef_euler_from_ned_c
from transformations cimport ned_euler_from_ecef as ned_euler_from_ecef_c


import cython
import numpy as np
cimport numpy as np

cdef np.ndarray[double, ndim=2] matrix2numpy(Matrix3 m):
    return np.array([
        [m(0, 0), m(0, 1), m(0, 2)],
        [m(1, 0), m(1, 1), m(1, 2)],
        [m(2, 0), m(2, 1), m(2, 2)],
    ])

cdef Matrix3 numpy2matrix (np.ndarray[double, ndim=2, mode="fortran"] m):
    assert m.shape[0] == 3
    assert m.shape[1] == 3
    return Matrix3(<double*>m.data)

cdef ECEF list2ecef(ecef):
    cdef ECEF e;
    e.x = ecef[0]
    e.y = ecef[1]
    e.z = ecef[2]
    return e

def euler2quat_single(euler):
    cdef Vector3 e = Vector3(euler[0], euler[1], euler[2])
    cdef Quaternion q = euler2quat_c(e)
    return [q.w(), q.x(), q.y(), q.z()]

def quat2euler_single(quat):
    cdef Quaternion q = Quaternion(quat[0], quat[1], quat[2], quat[3])
    cdef Vector3 e = quat2euler_c(q);
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
