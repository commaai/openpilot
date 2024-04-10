#!/usr/bin/env python3
import unittest

from openpilot.selfdrive.car.volkswagen.values import CAR


class TestVolkswagenPlatformConfigs(unittest.TestCase):
  def test_chassis_codes(self):
    for platform in CAR:
      with self.subTest(platform=platform):
        self.assertTrue(len(platform.config.chassis_codes) > 0, "Chassis codes not set")

        # No two platforms should share chassis codes
        for comp in CAR:
          if platform == comp:
            continue
          self.assertEqual(set(), platform.config.chassis_codes & comp.config.chassis_codes,
                           f"Shared chassis codes: {comp}")


if __name__ == "__main__":
  unittest.main()
