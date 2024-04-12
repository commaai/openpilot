#!/usr/bin/env python3
import re
import unittest

from cereal import car
from openpilot.selfdrive.car.volkswagen.values import CAR, FW_QUERY_CONFIG, WMI
from openpilot.selfdrive.car.volkswagen.fingerprints import FW_VERSIONS

Ecu = car.CarParams.Ecu

CHASSIS_CODE_PATTERN = re.compile('[A-Z0-9]{2}')
# TODO: determine the unknown groups
SPARE_PART_FW_PATTERN = re.compile(b'\xf1\x87(?P<gateway>[0-9][0-9A-Z]{2})(?P<unknown>[0-9][0-9A-Z][0-9])(?P<unknown2>[0-9A-Z]{2}[0-9])([A-Z0-9]| )')


class TestVolkswagenPlatformConfigs(unittest.TestCase):
  def test_spare_part_fw_pattern(self):
    # Relied on for determining if a FW is likely VW
    for platform, ecus in FW_VERSIONS.items():
      with self.subTest(platform=platform):
        for fws in ecus.values():
          for fw in fws:
            self.assertNotEqual(SPARE_PART_FW_PATTERN.match(fw), None, f"Bad FW: {fw}")

  def test_chassis_codes(self):
    for platform in CAR:
      with self.subTest(platform=platform):
        self.assertTrue(len(platform.config.wmis) > 0, "WMIs not set")
        self.assertTrue(len(platform.config.chassis_codes) > 0, "Chassis codes not set")
        self.assertTrue(all(CHASSIS_CODE_PATTERN.match(cc) for cc in
                            platform.config.chassis_codes), "Bad chassis codes")

        # No two platforms should share chassis codes
        for comp in CAR:
          if platform == comp:
            continue
          self.assertEqual(set(), platform.config.chassis_codes & comp.config.chassis_codes,
                           f"Shared chassis codes: {comp}")

  def test_custom_fuzzy_fingerprinting(self):
    for platform in CAR:
      expected_radar_fw = FW_VERSIONS[platform][Ecu.fwdRadar, 0x757, None]

      with self.subTest(platform=platform):
        for wmi in WMI:
          for chassis_code in platform.config.chassis_codes | {"00"}:
            vin = ["0"] * 17
            vin[0:3] = wmi
            vin[6:8] = chassis_code
            vin = "".join(vin)

            # Check a few FW cases - expected, unexpected
            for radar_fw in expected_radar_fw + [b'\xf1\x877H9907572AA\xf1\x890396']:
              should_match = ((wmi in platform.config.wmis and chassis_code in platform.config.chassis_codes) and
                              radar_fw in expected_radar_fw)

              live_fws = {(0x757, None): [radar_fw]}
              matches = FW_QUERY_CONFIG.match_fw_to_car_fuzzy(live_fws, vin, FW_VERSIONS)

              expected_matches = {platform} if should_match else set()
              self.assertEqual(expected_matches, matches, "Bad match")


if __name__ == "__main__":
  unittest.main()
