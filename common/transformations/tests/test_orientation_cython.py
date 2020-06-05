#!/usr/bin/env python3
# pylint: skip-file

import unittest

import numpy as np

from common.transformations.transformations import (euler2quat_single,
                                                    euler2rot_single,
                                                    quat2euler_single,
                                                    quat2rot_single,
                                                    rot2euler_single,
                                                    rot2quat_single)

eulers = np.array([[ 1.46520501,  2.78688383,  2.92780854],
                   [ 4.86909526,  3.60618161,  4.30648981],
                   [ 3.72175965,  2.68763705,  5.43895988],
                   [ 5.92306687,  5.69573614,  0.81100357],
                   [ 0.67838374,  5.02402037,  2.47106426]])

quats = np.array([[ 0.66855182, -0.71500939,  0.19539353,  0.06017818],
                  [ 0.43163717,  0.70013301,  0.28209145,  0.49389021],
                  [ 0.44121991, -0.08252646,  0.34257534,  0.82532207],
                  [ 0.88578382, -0.04515356, -0.32936046,  0.32383617],
                  [ 0.06578165,  0.61282835,  0.07126891,  0.78424163]])

ecef_positions = np.array([[-2711076.55270557, -4259167.14692758,  3884579.87669935],
                           [ 2068042.69652729, -5273435.40316622,  2927004.89190746],
                           [-2160412.60461669, -4932588.89873832,  3406542.29652851],
                           [-1458247.92550567,  5983060.87496612,  1654984.6099885 ],
                           [ 4167239.10867871,  4064301.90363223,  2602234.6065749 ]])

ned_eulers = np.array([[ 0.46806039, -0.4881889 ,  1.65697808],
                       [-2.14525969, -0.36533066,  0.73813479],
                       [-1.39523364, -0.58540761, -1.77376356],
                       [-1.84220435,  0.61828016, -1.03310421],
                       [ 2.50450101,  0.36304151,  0.33136365]])


class TestOrientationCython(unittest.TestCase):
  def test_euler2quat_quat2euler(self):
    """Euler angle representation is not unique, so they can't be compared directly.
    To test quat2euler we need to convert the euler angle back to quaternions and check with the input.
    euler2quat needs to be verified to be correct by a different test."""
    for quat in quats:
      euler = quat2euler_single(quat)
      quat_new = euler2quat_single(euler)
      np.testing.assert_allclose(quat, quat_new, rtol=1e-7)

  def test_euler2quat_single(self):
    for i in range(len(eulers)):
      np.testing.assert_allclose(quats[i], euler2quat_single(eulers[i]), rtol=1e-7)

  def test_quat2rot_rot2_quat(self):
    """Tests round trip of rotation matrix to quaternion conversion"""
    for quat in quats:
      rot = quat2rot_single(quat)
      quat_new = rot2quat_single(rot)
      np.testing.assert_allclose(quat, quat_new, rtol=1e-7)

  def test_euler2rot_rot2euler(self):
    """Tests euler to rotation matrix conversion. Converts to quaternions to verify results"""
    for euler in eulers:
      rot = euler2rot_single(euler)
      euler_new = rot2euler_single(rot)
      np.testing.assert_allclose(euler2quat_single(euler), euler2quat_single(euler_new), rtol=1e-7)

if __name__ == "__main__":
  unittest.main()
