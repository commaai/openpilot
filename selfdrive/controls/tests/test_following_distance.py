#!/usr/bin/env python3
import unittest
import numpy as np

from selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import desired_follow_distance
from selfdrive.test.longitudinal_maneuvers.maneuver import Maneuver

def run_following_distance_simulation(v_lead, t_end=100.0):
  man = Maneuver(
    '',
    duration=t_end,
    initial_speed=float(v_lead),
    lead_relevancy=True,
    initial_distance_lead=100,
    speed_lead_values=[v_lead],
    breakpoints=[0.],
  )
  valid, output = man.evaluate()
  assert valid
  return output[-1,2] - output[-1,1]


class TestFollowingDistance(unittest.TestCase):
  def test_following_distanc(self):
    for speed in np.arange(0, 40, 5):
      print(f'Testing {speed} m/s')
      v_lead = float(speed)

      simulation_steady_state = run_following_distance_simulation(v_lead)
      correct_steady_state = desired_follow_distance(v_lead, v_lead)

      self.assertAlmostEqual(simulation_steady_state, correct_steady_state, delta=(correct_steady_state*.1 + .5))


if __name__ == "__main__":
  unittest.main()
