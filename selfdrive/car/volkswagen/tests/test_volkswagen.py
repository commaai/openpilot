#!/usr/bin/env python3
import re
import unittest

from cereal import car
from openpilot.selfdrive.car.volkswagen.values import CAR, SPARE_PART_FW_PATTERN, FW_QUERY_CONFIG, GATEWAY_TYPES, WMI
from openpilot.selfdrive.car.volkswagen.fingerprints import FW_VERSIONS

Ecu = car.CarParams.Ecu
CHASSIS_CODE_PATTERN = re.compile('[A-Z0-9]{2}')


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
        self.assertTrue(len(platform.config.chassis_codes) > 0, "Chassis codes not set")
        self.assertTrue(all(CHASSIS_CODE_PATTERN.match(cc) for cc in
                            platform.config.chassis_codes), "Bad chassis codes")

        # No two platforms should share chassis codes
        for comp in CAR:
          if platform == comp:
            continue
          self.assertEqual(set(), platform.config.chassis_codes & comp.config.chassis_codes,
                           f"Shared chassis codes: {comp}")

  def test_custom_fingerprinting(self):
    for platform in CAR:
      with self.subTest(platform=platform):
        for wmi in WMI | {"000"}:
          for chassis_code in platform.config.chassis_codes | {"00"}:
            vin = ["0"] * 17
            vin[0:3] = wmi
            vin[6:8] = chassis_code
            vin = "".join(vin)

            # Check a few FW gateway type cases - expected, unexpected, no match
            for radar_fw in (
              b'\xf1\x872Q0907572AA\xf1\x890396',
              b'\xf1\x877H9907572AA\xf1\x890396',
              b'',
            ):
              match = SPARE_PART_FW_PATTERN.match(radar_fw)
              should_match = (wmi != "000" and chassis_code != "00" and
                              match is not None and match.group("gateway") in GATEWAY_TYPES[(Ecu.fwdRadar, 0x757, None)])

              live_fws = {(0x757, None): [radar_fw]}
              _, matches = FW_QUERY_CONFIG.match_fw_to_car(live_fws, vin, {})

              expected_matches = {platform} if should_match else set()
              self.assertEqual(expected_matches, matches, "Bad match")


if __name__ == "__main__":
  unittest.main()
