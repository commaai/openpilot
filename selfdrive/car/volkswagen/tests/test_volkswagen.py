#!/usr/bin/env python3
import unittest

from openpilot.selfdrive.car.volkswagen.values import CAR


class TestVolkswagenPlatformConfigs(unittest.TestCase):
  def test_configs(self):
    for platform in CAR:
      self.assertTrue(len(platform.config.chassis_codes) > 0, "Chassis code not set")


if __name__ == "__main__":
  unittest.main()
