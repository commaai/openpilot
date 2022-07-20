#!/usr/bin/env python3
import unittest
from unittest.mock import Mock, patch
from parameterized import parameterized

from selfdrive.thermald.fan_controller import TiciFanController

ALL_CONTROLLERS = [(TiciFanController,)]

def patched_controller(controller_class):
  with patch("os.system", new=Mock()):
    return controller_class()

class TestFanController(unittest.TestCase):
  def wind_up(self, controller, ignition=True):
    for _ in range(1000):
      controller.update(max_cpu_temp=100, ignition=ignition)

  def wind_down(self, controller, ignition=False):
    for _ in range(1000):
      controller.update(max_cpu_temp=10, ignition=ignition)

  @parameterized.expand(ALL_CONTROLLERS)
  def test_hot_onroad(self, controller_class):
    controller = patched_controller(controller_class)
    self.wind_up(controller)
    self.assertGreaterEqual(controller.update(max_cpu_temp=100, ignition=True), 70)

  @parameterized.expand(ALL_CONTROLLERS)
  def test_offroad_limits(self, controller_class):
    controller = patched_controller(controller_class)
    self.wind_up(controller)
    self.assertLessEqual(controller.update(max_cpu_temp=100, ignition=False), 30)

  @parameterized.expand(ALL_CONTROLLERS)
  def test_no_fan_wear(self, controller_class):
    controller = patched_controller(controller_class)
    self.wind_down(controller)
    self.assertEqual(controller.update(max_cpu_temp=10, ignition=False), 0)

  @parameterized.expand(ALL_CONTROLLERS)
  def test_limited(self, controller_class):
    controller = patched_controller(controller_class)
    self.wind_up(controller, ignition=True)
    self.assertGreaterEqual(controller.update(max_cpu_temp=100, ignition=True), 80)

  @parameterized.expand(ALL_CONTROLLERS)
  def test_windup_speed(self, controller_class):
    controller = patched_controller(controller_class)
    self.wind_down(controller, ignition=True)
    for _ in range(10):
      controller.update(max_cpu_temp=90, ignition=True)
    self.assertGreaterEqual(controller.update(max_cpu_temp=90, ignition=True), 60)

if __name__ == "__main__":
  unittest.main()
