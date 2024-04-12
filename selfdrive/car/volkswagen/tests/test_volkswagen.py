#!/usr/bin/env python3
import re
import unittest

from cereal import car
from openpilot.selfdrive.car.volkswagen.values import CAR, FW_QUERY_CONFIG, WMI, get_gateway_types
from openpilot.selfdrive.car.volkswagen.fingerprints import FW_VERSIONS

Ecu = car.CarParams.Ecu
CHASSIS_CODE_PATTERN = re.compile('[A-Z0-9]{2}')


class TestVolkswagenPlatformConfigs(unittest.TestCase):
  def test_spare_part_fw_pattern(self):
    # Relied on for determining if a FW is likely VW
    results = get_gateway_types([
      b'\xf1\x872Q0907572AA\xf1\x890396',
      b'\xf1\x873Q0907572C \xf1\x890195',
      b'\xf1\x875Q0907567M \xf1\x890398\xf1\x82\x0101',
    ])
    self.assertEqual(results, {b"2Q0", b"3Q0", b"5Q0"})

    # All should be parsable
    for platform, ecus in FW_VERSIONS.items():
      with self.subTest(platform=platform):
        for fws in ecus.values():
          for fw in fws:
            gateway_types = get_gateway_types([fw])
            self.assertEqual(len(gateway_types), 1, f"Bad FW: {fw}")

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

  def test_custom_fuzzy_fingerprinting(self):
    for platform in CAR:
      expected_gateway_types = get_gateway_types(FW_VERSIONS[platform][Ecu.fwdRadar, 0x757, None])
      expected_radar_fw = FW_VERSIONS[platform][Ecu.fwdRadar, 0x757, None]

      with self.subTest(platform=platform):
        for wmi in WMI:
          for chassis_code in platform.config.chassis_codes | {"00"}:
            vin = ["0"] * 17
            vin[0:3] = wmi
            vin[6:8] = chassis_code
            vin = "".join(vin)

            # Check a few FW gateway type cases - expected, unexpected, no match
            for radar_fw in expected_radar_fw + [
              b'\xf1\x877H9907572AA\xf1\x890396',
              b'',
            ]:
              found_gateway_types = get_gateway_types([radar_fw])
              should_match = ((wmi in platform.config.wmis and chassis_code in platform.config.chassis_codes) and
                              len(found_gateway_types) and list(found_gateway_types)[0] in expected_gateway_types)

              live_fws = {(0x757, None): [radar_fw]}
              matches = FW_QUERY_CONFIG.match_fw_to_car_fuzzy(live_fws, vin, FW_VERSIONS)

              expected_matches = {platform} if should_match else set()
              self.assertEqual(expected_matches, matches, "Bad match")


if __name__ == "__main__":
  unittest.main()
