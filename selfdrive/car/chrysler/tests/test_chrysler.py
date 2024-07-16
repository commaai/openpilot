import random
import re

from cereal import car
from openpilot.selfdrive.car.chrysler.values import CAR, FW_QUERY_CONFIG
from openpilot.selfdrive.car.chrysler.fingerprints import FW_VERSIONS

Ecu = car.CarParams.Ecu

CHASSIS_CODE_PATTERN = re.compile('[A-Z0-9]{2}')


class TestChryslerPlatformConfigs:
  def test_vin_parse_codes(self, subtests):
    for platform in CAR:
      with subtests.test(platform=platform.value):
        assert len(platform.config.chassis_codes) > 0, "Chassis codes not set"
        assert len(platform.config.years) > 0, "years codes not set"
        assert all(CHASSIS_CODE_PATTERN.match(cc) for cc in \
                            platform.config.chassis_codes), "Bad chassis codes"

        # No two platforms should share chassis codes
        for comp in CAR:
          if platform == comp:
            continue
          assert set() == (platform.config.chassis_codes & comp.config.chassis_codes) \
                              or (platform.config.years != comp.config.years), \
                           f"Shared chassis codes: {comp}"

  def test_custom_fuzzy_fingerprinting(self, subtests):
    all_radar_fw = list({fw for ecus in FW_VERSIONS.values() for fw in ecus[Ecu.fwdRadar, 0x753, None]})

    for platform in CAR:
      with subtests.test(platform=platform.name):
        for year in platform.config.years:
          for chassis_code in platform.config.chassis_codes | {"00"}:
            vin = ["0"] * 17
            vin[9] = year
            vin[4:6] = chassis_code
            vin = "".join(vin)

            # Check a few FW cases - expected, unexpected
            for radar_fw in random.sample(all_radar_fw, 5) + [b'\xf1\x875Q0907572G \xf1\x890571', b'\xf1\x877H9907572AA\xf1\x890396']:

              should_match = ((chassis_code in platform.config.chassis_codes and \
                              platform.config.years[0] <= year <= platform.config.years[1]) and radar_fw in all_radar_fw)

              live_fws = {(0x753, None): [radar_fw]}
              matches = FW_QUERY_CONFIG.match_fw_to_car_fuzzy(live_fws, vin, FW_VERSIONS)

              expected_matches = {platform} if should_match else set()
              assert expected_matches == matches, "Bad match"