#!/usr/bin/env python3
import unittest
import numpy as np
import math

from cereal import car
from common.conversions import Conversions as CV
from selfdrive.controls.lib.drive_helpers import update_v_cruise, CRUISE_LONG_PRESS, \
                                                 CRUISE_LONG_PRESS_ERROR, V_CRUISE_MIN, V_CRUISE_MAX

ButtonEvent = car.CarState.ButtonEvent
ButtonType = car.CarState.ButtonEvent.Type

TEST_KPH_STEP = 0.01


class TestCruiseButtons(unittest.TestCase):
  enabled = True
  metric = False
  v_cruise_delta = 1. if metric else 1.6

  @staticmethod
  def _get_button_timers(accel=0, decel=0):
    return {ButtonEvent.Type.accelCruise: accel, ButtonEvent.Type.decelCruise: decel}

  @staticmethod
  def _get_button_events(accel=False, decel=False):
    # Returns unpressed cruise button events
    button_events = []
    if accel:
      be = car.CarState.ButtonEvent(type=ButtonType.accelCruise)
      button_events.append(be)
    if decel:
      be = car.CarState.ButtonEvent(type=ButtonType.decelCruise)
      button_events.append(be)
    return button_events

  # TODO: DONE
  # def test_v_cruise_limits(self):
  #   # Test minimum limit
  #   button_timers = self._get_button_timers(decel=1)
  #   button_events = self._get_button_events(decel=True)
  #   v_cruise_kph = update_v_cruise(V_CRUISE_MIN, button_events, button_timers, enabled=True, metric=self.metric)
  #   self.assertEqual(v_cruise_kph, V_CRUISE_MIN)
  #
  #   # Test maximum limit
  #   button_timers = self._get_button_timers(accel=1)
  #   button_events = self._get_button_events(accel=True)
  #   v_cruise_kph = update_v_cruise(V_CRUISE_MAX, button_events, button_timers, enabled=True, metric=self.metric)
  #   self.assertEqual(v_cruise_kph, V_CRUISE_MAX)

  # TODO: DONE
  # def test_button_accel(self):
  #   button_events = self._get_button_events(accel=True)
  #   button_timers = self._get_button_timers(accel=1)
  #   for speed in np.arange(V_CRUISE_MIN, V_CRUISE_MAX + TEST_KPH_STEP, TEST_KPH_STEP):
  #     v_cruise_kph = update_v_cruise(speed, button_events, button_timers, enabled=True, metric=self.metric)
  #     expected_v_cruise_kph = min(V_CRUISE_MAX, round(speed + v_cruise_delta, 1))
  #     self.assertEqual(v_cruise_kph, expected_v_cruise_kph)

  # TODO: DONE
  # def test_button_decel(self):
  #   button_events = self._get_button_events(decel=True)
  #   button_timers = self._get_button_timers(decel=1)
  #   for speed in np.arange(V_CRUISE_MIN, V_CRUISE_MAX + TEST_KPH_STEP, TEST_KPH_STEP):
  #     v_cruise_kph = update_v_cruise(speed, button_events, button_timers, enabled=True, metric=self.metric)
  #     expected_v_cruise_kph = max(V_CRUISE_MIN, round(speed - v_cruise_delta, 1))
  #     self.assertEqual(v_cruise_kph, expected_v_cruise_kph)

  def test_button_accel_long_press(self):
    button_events = self._get_button_events(accel=False)
    button_timers = self._get_button_timers(accel=CRUISE_LONG_PRESS)
    v_cruise_delta = self.v_cruise_delta * 5
    for speed in np.arange(V_CRUISE_MIN, V_CRUISE_MAX + TEST_KPH_STEP, TEST_KPH_STEP):
      closest_interval = round(speed / v_cruise_delta) * v_cruise_delta
      if abs(closest_interval - speed) > CRUISE_LONG_PRESS_ERROR:  # partial interval
        expected_v_cruise_kph = math.ceil(speed / v_cruise_delta) * v_cruise_delta
      else:
        expected_v_cruise_kph = speed + v_cruise_delta
      v_cruise_kph = update_v_cruise(speed, button_events, button_timers, self.enabled, self.metric)
      self.assertEqual(v_cruise_kph, min(V_CRUISE_MAX, expected_v_cruise_kph))

  def test_button_decel_long_press(self):
    pass

  # TODO: test long press entrance conditions


if __name__ == "__main__":
  unittest.main()
