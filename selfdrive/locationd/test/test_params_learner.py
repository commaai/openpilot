#!/usr/bin/env python

import numpy as np
import unittest

from selfdrive.car.honda.interface import CarInterface
from selfdrive.car.honda.values import CAR
from selfdrive.controls.lib.vehicle_model import VehicleModel
from selfdrive.locationd.liblocationd_py import liblocationd  # pylint: disable=no-name-in-module, import-error


class TestParamsLearner(unittest.TestCase):
  def setUp(self):

    self.CP = CarInterface.get_params(CAR.CIVIC, {})
    bts = self.CP.to_bytes()

    self.params_learner = liblocationd.params_learner_init(len(bts), bts, 0.0, 1.0, self.CP.steerRatio, 1.0)

  def test_convergence(self):
    # Setup vehicle model with wrong parameters
    VM_sim = VehicleModel(self.CP)
    x_target = 0.75
    sr_target = self.CP.steerRatio - 0.5
    ao_target = -1.0
    VM_sim.update_params(x_target, sr_target)

    # Run simulation
    times = np.arange(0, 15*3600, 0.01)
    angle_offset = np.radians(ao_target)
    steering_angles = np.radians(10 * np.sin(2 * np.pi * times / 100.)) + angle_offset
    speeds = 10 * np.sin(2 * np.pi * times / 1000.) + 25

    for i, t in enumerate(times):
      u = speeds[i]
      sa = steering_angles[i]
      psi = VM_sim.yaw_rate(sa - angle_offset, u)
      liblocationd.params_learner_update(self.params_learner, psi, u, sa)

    # Verify learned parameters
    sr = liblocationd.params_learner_get_sR(self.params_learner)
    ao_slow = np.degrees(liblocationd.params_learner_get_slow_ao(self.params_learner))
    x = liblocationd.params_learner_get_x(self.params_learner)
    self.assertAlmostEqual(x_target, x, places=1)
    self.assertAlmostEqual(ao_target, ao_slow, places=1)
    self.assertAlmostEqual(sr_target, sr, places=1)





if __name__ == "__main__":
  unittest.main()
