#!/usr/bin/env python3
import numpy as np
from parameterized import parameterized_class
import unittest

from selfdrive.controls.lib.drive_helpers import VCruiseHelper, V_CRUISE_MAX, V_CRUISE_ENABLE_MIN
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
  return output[-1, 3]


class TestCruiseSpeed(unittest.TestCase):
  def test_cruise_speed(self):
    params = Params()
    for e2e in [False, True]:
      params.put_bool("ExperimentalMode", e2e)
      for speed in np.arange(5, 40, 5):
        print(f'Testing {speed} m/s')
        cruise_speed = float(speed)

        simulation_steady_state = run_cruise_simulation(cruise_speed)
        self.assertAlmostEqual(simulation_steady_state, cruise_speed, delta=.01, msg=f'Did not reach {speed} m/s')


# TODO: test pcmCruise
@parameterized_class(('pcm_cruise',), [(False,)])
class TestVCruiseHelper(unittest.TestCase):
  def setUp(self):
    self.CP = car.CarParams(pcmCruise=self.pcm_cruise)  # pylint: disable=E1101
    self.v_cruise_helper = VCruiseHelper(self.CP)

  def test_adjust_speed(self):
    """
    Asserts speed changes on falling edges of buttons.
    """

    self.v_cruise_helper.initialize_v_cruise(car.CarState())

    for btn in (ButtonType.accelCruise, ButtonType.decelCruise):
      initial_v_cruise = self.v_cruise_helper.v_cruise_kph
      for pressed in (True, False):
        CS = car.CarState(cruiseState={"available": True})
        CS.buttonEvents = [ButtonEvent(type=btn, pressed=pressed)]

        self.v_cruise_helper.update_v_cruise(CS, enabled=True, is_metric=False)
        self.assertEqual(pressed, (initial_v_cruise == self.v_cruise_helper.v_cruise_kph))

  def test_resume_in_standstill(self):
    """
    Asserts we don't increment set speed if user presses resume/accel to exit cruise standstill.
    """

    self.v_cruise_helper.initialize_v_cruise(car.CarState())
    initial_v_cruise = self.v_cruise_helper.v_cruise_kph

    for standstill in (True, False):
      for pressed in (True, False):
        CS = car.CarState(cruiseState={"available": True, "standstill": standstill})
        CS.buttonEvents = [ButtonEvent(type=ButtonType.accelCruise, pressed=pressed)]

        self.v_cruise_helper.update_v_cruise(CS, enabled=True, is_metric=False)
        # speed should only update if not at standstill and button falling edge
        should_equal = standstill or pressed
        self.assertEqual(should_equal, (initial_v_cruise == self.v_cruise_helper.v_cruise_kph))

  def test_initialize_v_cruise(self):
    """
    Asserts allowed cruise speeds on enabling with SET
    """

    for v_ego in np.linspace(0, 100, 101):
      self.v_cruise_helper.initialize_v_cruise(car.CarState(vEgo=float(v_ego)))
      self.assertTrue(V_CRUISE_ENABLE_MIN <= self.v_cruise_helper.v_cruise_kph <= V_CRUISE_MAX)
      self.assertTrue(self.v_cruise_helper.v_cruise_initialized)


if __name__ == "__main__":
  unittest.main()
