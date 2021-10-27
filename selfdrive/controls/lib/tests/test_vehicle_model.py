#!/usr/bin/env python3
import math
import unittest

import numpy as np

from selfdrive.car.honda.interface import CarInterface
from selfdrive.car.honda.values import CAR
from selfdrive.controls.lib.vehicle_model import VehicleModel


class TestVehicleModel(unittest.TestCase):
  def setUp(self):
    CP = CarInterface.get_params(CAR.CIVIC)
    self.VM = VehicleModel(CP)

  def test_round_trip_yaw_rate(self):
    # TODO: fix VM to work at zero speed
    for u in np.linspace(1, 30, num=10):
      for roll in np.linspace(math.radians(-20), math.radians(20), num=11):
        for sa in np.linspace(math.radians(-20), math.radians(20), num=11):
          yr = self.VM.yaw_rate(sa, u, roll)
          new_sa = self.VM.get_steer_from_yaw_rate(yr, u, roll)

          self.assertAlmostEqual(sa, new_sa)


if __name__ == "__main__":
  unittest.main()
