#!/usr/bin/env python3
from unittest.mock import Mock, patch
from parameterized import parameterized

from openpilot.selfdrive.thermald.fan_controller import TiciFanController

ALL_CONTROLLERS = [(TiciFanController,)]

def patched_controller(controller_class):
  with patch("os.system", new=Mock()):
    return controller_class()

class TestFanController:
  def wind_up(self, controller, ignition=True):
    for _ in range(1000):
      controller.update(100, ignition)

  def wind_down(self, controller, ignition=False):
    for _ in range(1000):
      controller.update(10, ignition)

  @parameterized.expand(ALL_CONTROLLERS)
  def test_hot_onroad(self, controller_class):
    controller = patched_controller(controller_class)
    self.wind_up(controller)
    assert controller.update(100, True) >= 70

  @parameterized.expand(ALL_CONTROLLERS)
  def test_offroad_limits(self, controller_class):
    controller = patched_controller(controller_class)
    self.wind_up(controller)
    assert controller.update(100, False) <= 30

  @parameterized.expand(ALL_CONTROLLERS)
  def test_no_fan_wear(self, controller_class):
    controller = patched_controller(controller_class)
    self.wind_down(controller)
    assert controller.update(10, False) == 0

  @parameterized.expand(ALL_CONTROLLERS)
  def test_limited(self, controller_class):
    controller = patched_controller(controller_class)
    self.wind_up(controller, True)
    assert controller.update(100, True) == 100

  @parameterized.expand(ALL_CONTROLLERS)
  def test_windup_speed(self, controller_class):
    controller = patched_controller(controller_class)
    self.wind_down(controller, True)
    for _ in range(10):
      controller.update(90, True)
    assert controller.update(90, True) >= 60
