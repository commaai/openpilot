#!/usr/bin/env python3

import numpy as np
import unittest

from openpilot.common.transformations.orientation import euler2quat, quat2euler, euler2rot, rot2euler, \
                                               rot2quat, quat2rot, \
                                               ned_euler_from_ecef

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


class TestOrientation(unittest.TestCase):
  def test_quat_euler(self):
    for i, eul in enumerate(eulers):
      np.testing.assert_allclose(quats[i], euler2quat(eul), rtol=1e-7)
      np.testing.assert_allclose(quats[i], euler2quat(quat2euler(quats[i])), rtol=1e-6)
    for i, eul in enumerate(eulers):
      np.testing.assert_allclose(quats[i], euler2quat(list(eul)), rtol=1e-7)
      np.testing.assert_allclose(quats[i], euler2quat(quat2euler(list(quats[i]))), rtol=1e-6)
    np.testing.assert_allclose(quats, euler2quat(eulers), rtol=1e-7)
    np.testing.assert_allclose(quats, euler2quat(quat2euler(quats)), rtol=1e-6)

  def test_rot_euler(self):
    for eul in eulers:
      np.testing.assert_allclose(euler2quat(eul), euler2quat(rot2euler(euler2rot(eul))), rtol=1e-7)
    for eul in eulers:
      np.testing.assert_allclose(euler2quat(eul), euler2quat(rot2euler(euler2rot(list(eul)))), rtol=1e-7)
    np.testing.assert_allclose(euler2quat(eulers), euler2quat(rot2euler(euler2rot(eulers))), rtol=1e-7)

  def test_rot_quat(self):
    for quat in quats:
      np.testing.assert_allclose(quat, rot2quat(quat2rot(quat)), rtol=1e-7)
    for quat in quats:
      np.testing.assert_allclose(quat, rot2quat(quat2rot(list(quat))), rtol=1e-7)
    np.testing.assert_allclose(quats, rot2quat(quat2rot(quats)), rtol=1e-7)

  def test_euler_ned(self):
    for i in range(len(eulers)):
      np.testing.assert_allclose(ned_eulers[i], ned_euler_from_ecef(ecef_positions[i], eulers[i]), rtol=1e-7)
      #np.testing.assert_allclose(eulers[i], ecef_euler_from_ned(ecef_positions[i], ned_eulers[i]), rtol=1e-7)
    # np.testing.assert_allclose(ned_eulers, ned_euler_from_ecef(ecef_positions, eulers), rtol=1e-7)


if __name__ == "__main__":
  unittest.main()
