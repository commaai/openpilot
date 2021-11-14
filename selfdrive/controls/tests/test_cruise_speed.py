#!/usr/bin/env python3
import unittest
import numpy as np
from selfdrive.test.longitudinal_maneuvers.maneuver import Maneuver

def run_cruise_simulation(cruise, t_end=100.):
  man = Maneuver(
    '',
    duration=t_end,
    initial_speed=float(0.),
    lead_relevancy=True,
    initial_distance_lead=100,
    cruise_values=[cruise],
    prob_lead_values=[0.0],
    breakpoints=[0.],
  )
  valid, output = man.evaluate()
  assert valid
  return output[-1,3]


class TestCruiseSpeed(unittest.TestCase):
  def test_cruise_speed(self):
    for speed in np.arange(5, 40, 5):
      print(f'Testing {speed} m/s')
      cruise_speed = float(speed)

      simulation_steady_state = run_cruise_simulation(cruise_speed)
      self.assertAlmostEqual(simulation_steady_state, cruise_speed, delta=.01, msg=f'Did not reach {speed} m/s')


if __name__ == "__main__":
  unittest.main()
