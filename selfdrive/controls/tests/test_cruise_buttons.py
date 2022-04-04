#!/usr/bin/env python3
import unittest
import numpy as np
from common.numpy_fast import clip

from cereal import car
from common.conversions import Conversions as CV
from selfdrive.controls.lib.drive_helpers import update_v_cruise, CRUISE_LONG_PRESS, \
                                                 V_CRUISE_MIN, V_CRUISE_MAX

ButtonEvent = car.CarState.ButtonEvent
ButtonType = car.CarState.ButtonEvent.Type

TEST_KPH_STEP = 0.01


def clip_mph_speed(speed):
  return clip(speed, V_CRUISE_MIN * CV.KPH_TO_MPH, V_CRUISE_MAX * CV.KPH_TO_MPH)


class TestCruiseButtons(unittest.TestCase):
  enabled = True
  metric = False
  v_cruise_delta = 1. if metric else CV.MPH_TO_KPH

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

  def test_accel_long_press(self):
    button_timers = self._get_button_timers(accel=CRUISE_LONG_PRESS)
    for speed in np.arange(0, V_CRUISE_MAX * CV.KPH_TO_MPH, 5):
      expected_speed_mph = clip_mph_speed(speed + 5)
      v_cruise_kph = update_v_cruise(speed * CV.MPH_TO_KPH, [], button_timers, self.enabled, self.metric)
      self.assertAlmostEqual(expected_speed_mph, v_cruise_kph * CV.KPH_TO_MPH, delta=0.01)

  def test_accel(self):
    button_events = self._get_button_events(accel=True)
    button_timers = self._get_button_timers(accel=1)
    for speed in np.arange(0, V_CRUISE_MAX * CV.KPH_TO_MPH, 5):
      v_cruise_kph = update_v_cruise(speed * CV.MPH_TO_KPH, button_events, button_timers, self.enabled, self.metric)
      expected_speed_mph = clip_mph_speed(speed + 1)
      self.assertAlmostEqual(expected_speed_mph, v_cruise_kph * CV.KPH_TO_MPH, delta=0.01)


if __name__ == "__main__":
  unittest.main()
