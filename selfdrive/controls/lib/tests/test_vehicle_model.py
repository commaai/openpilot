#!/usr/bin/env python3
import math
import unittest

import numpy as np
from control import StateSpace

from selfdrive.car.honda.interface import CarInterface
from selfdrive.car.honda.values import CAR
from selfdrive.controls.lib.vehicle_model import VehicleModel, dyn_ss_sol, create_dyn_state_matrices


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

  def test_dyn_ss_sol_against_yaw_rate(self):
    """Verify that the yaw_rate helper function matches the results
    from the state space model."""

    for roll in np.linspace(math.radians(-20), math.radians(20), num=11):
      for u in np.linspace(1, 30, num=10):
        for sa in np.linspace(math.radians(-20), math.radians(20), num=11):

          # Compute yaw rate based on state space model
          _, yr1 = dyn_ss_sol(sa, u, roll, self.VM)

          # Compute yaw rate using direct computations
          yr2 = self.VM.yaw_rate(sa, u, roll)
          self.assertAlmostEqual(float(yr1), yr2)

  def test_syn_ss_sol_simulate(self):
    """Verifies that dyn_ss_sol mathes a simulation"""

    for roll in np.linspace(math.radians(-20), math.radians(20), num=11):
      for u in np.linspace(1, 30, num=10):
        A, B = create_dyn_state_matrices(u, self.VM)

        # Convert to discrete time system
        ss = StateSpace(A, B, np.eye(2), np.zeros((2, 2)))
        ss = ss.sample(0.01)

        for sa in np.linspace(math.radians(-20), math.radians(20), num=11):
          inp = np.array([[sa], [roll]])

          # Simulate for 1 second
          x1 = np.zeros((2, 1))
          for _ in range(100):
            x1 = ss.A @ x1 + ss.B @ inp

          # Compute steady state solution directly
          x2 = dyn_ss_sol(sa, u, roll, self.VM)

          np.testing.assert_almost_equal(x1, x2, decimal=3)



if __name__ == "__main__":
  unittest.main()
