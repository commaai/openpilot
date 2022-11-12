#!/usr/bin/env python3
import numpy as np
from parameterized import parameterized_class
import unittest

from selfdrive.controls.lib.drive_helpers import VCruiseHelper
from cereal import car
from common.params import Params
from selfdrive.test.longitudinal_maneuvers.maneuver import Maneuver


ButtonEvent = car.CarState.ButtonEvent
ButtonType = car.CarState.ButtonEvent.Type


def run_cruise_simulation(cruise, t_end=20.):
  man = Maneuver(
    '',
    duration=t_end,
    initial_speed=max(cruise - 1., 0.0),
    lead_relevancy=True,
    initial_distance_lead=100,
    cruise_values=[cruise],
    prob_lead_values=[0.0],
    breakpoints=[0.],
  )
  valid, output = man.evaluate()
  assert valid
  return output[-1,3]


# class TestCruiseSpeed(unittest.TestCase):
#   def test_cruise_speed(self):
#     params = Params()
#     for e2e in [False, True]:
#       params.put_bool("ExperimentalMode", e2e)
#       for speed in np.arange(5, 40, 5):
#         print(f'Testing {speed} m/s')
#         cruise_speed = float(speed)
#
#         simulation_steady_state = run_cruise_simulation(cruise_speed)
#         self.assertAlmostEqual(simulation_steady_state, cruise_speed, delta=.01, msg=f'Did not reach {speed} m/s')


@parameterized_class(('pcm_cruise',), [(True,), (False,)])
class TestVCruiseHelper(unittest.TestCase):
  def setUp(self):
    self.CP = car.CarParams(pcmCruise=self.pcm_cruise)
    self.v_cruise_helper = VCruiseHelper(self.CP)

  def test_v_cruise_initialized_pcm(self):
    if not self.CP.pcmCruise:
      raise unittest.SkipTest

    for cruise_available in [False, True, False]:
      CS = car.CarState(cruiseState={"available": cruise_available, "speed": 10.})
      self.v_cruise_helper.update_v_cruise(CS, False, False)
      self.assertEqual(cruise_available, self.v_cruise_helper.v_cruise_initialized)


if __name__ == "__main__":
  unittest.main()
