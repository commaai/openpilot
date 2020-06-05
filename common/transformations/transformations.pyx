from transformations cimport Matrix3, Vector3, Quaternion
from transformations cimport ECEF, NED, Geodetic

from transformations cimport euler2quat as euler2quat_c
from transformations cimport quat2euler as quat2euler_c

import numpy as np

def euler2quat_single(euler):
    cdef Vector3 e = Vector3(euler[0], euler[1], euler[2])
    cdef Quaternion q = euler2quat_c(e)
    return [q.w(), q.x(), q.y(), q.z()]


def quat2euler_single(quat):
    cdef Quaternion q = Quaternion(quat[0], quat[1], quat[2], quat[3])
    cdef Vector3 e = quat2euler_c(q);
    return [e(0), e(1), e(2)]
