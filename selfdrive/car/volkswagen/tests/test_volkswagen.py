#!/usr/bin/env python3
import unittest

from openpilot.selfdrive.car.volkswagen.values import CAR, FW_QUERY_CONFIG


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

  def test_custom_fingerprinting(self):
    matches = FW_QUERY_CONFIG.match_fw_to_car_custom(None, '0' * 17, None)
    self.assertEqual(set(), matches, "Bad match")

    for platform in CAR:
      with self.subTest(platform=platform):
        for chassis_code in platform.config.chassis_codes:
          vin = ['0'] * 17
          vin[0:3] = 'WVW'
          vin[6:8] = chassis_code
          vin = ''.join(vin)

          matches = FW_QUERY_CONFIG.match_fw_to_car_custom(None, vin, None)
          self.assertEqual({platform}, matches, "Bad match")


if __name__ == "__main__":
  unittest.main()
