#!/usr/bin/env python3

import unittest

import numpy as np

from common.transformations.orientation import (ecef_euler_from_ned_single,
                                                euler2quat_single,
                                                euler2rot_single,
                                                ned_euler_from_ecef_single,
                                                quat2euler_single,
                                                quat2rot_single,
                                                rot2euler_single,
                                                rot2quat_single, rot_matrix)
from common.transformations.tests.test_orientation import eulers, quats, ecef_positions, ned_eulers  #


def rot_matrix_py(roll, pitch, yaw):
  cr, sr = np.cos(roll), np.sin(roll)
  cp, sp = np.cos(pitch), np.sin(pitch)
  cy, sy = np.cos(yaw), np.sin(yaw)
  rr = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
  rp = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
  ry = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
  return ry.dot(rp.dot(rr))


class TestOrientationCython(unittest.TestCase):
  def test_euler2quat_quat2euler(self):
    """Euler angle representation is not unique, so they can't be compared directly.
    To test quat2euler we need to convert the euler angle back to quaternions and check with the input.
    euler2quat needs to be verified to be correct by a different test."""
    for quat in quats:
      np.testing.assert_allclose(quat, euler2quat_single(quat2euler_single(quat)), rtol=1e-7)

  def test_euler2quat(self):
    for i in range(len(eulers)):
      np.testing.assert_allclose(quats[i], euler2quat_single(eulers[i]), rtol=1e-7)

  def test_quat2rot_rot2_quat(self):
    """Tests round trip of rotation matrix to quaternion conversion"""
    for i in range(len(quats)):
      np.testing.assert_allclose(quats[i], rot2quat_single(quat2rot_single(quats[i])), rtol=1e-7)

  def test_euler2rot_rot2euler(self):
    """Tests round trip of rotation matrix to euler conversion. Converts to quaternions to verify results"""
    for euler in eulers:
      rot = euler2rot_single(euler)
      euler_new = rot2euler_single(rot)
      np.testing.assert_allclose(euler2quat_single(euler), euler2quat_single(euler_new), rtol=1e-7)

  def test_rot2euler_quat2rot(self):
    for quat in quats:
      rot = quat2rot_single(quat)
      euler = rot2euler_single(rot)
      quat_new = euler2quat_single(euler)
      np.testing.assert_allclose(quat, quat_new, rtol=1e-7)

  def test_rot_matrix(self):
    for euler in eulers:
      roll, pitch, yaw = euler
      rot = rot_matrix(roll, pitch, yaw)
      np.testing.assert_allclose(rot_matrix_py(roll, pitch, yaw), rot, rtol=1e-7)

  def test_ecef_euler_from_ned(self):
    for i in range(len(eulers)):
      np.testing.assert_allclose(euler2quat_single(eulers[i]),
                                 euler2quat_single(ecef_euler_from_ned_single(ecef_positions[i], ned_eulers[i])),
                                 rtol=1e-7)

  def test_ned_euler_from_ecef(self):
    for i in range(len(eulers)):
      np.testing.assert_allclose(euler2quat_single(ned_eulers[i]),
                                 euler2quat_single(ned_euler_from_ecef_single(ecef_positions[i], eulers[i])), rtol=1e-7)


if __name__ == "__main__":
  unittest.main()
