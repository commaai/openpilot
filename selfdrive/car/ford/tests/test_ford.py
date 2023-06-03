#!/usr/bin/env python3
from collections import defaultdict
import unittest

from cereal import car
from selfdrive.car.ford.values import CAR, FW_QUERY_CONFIG, FW_VERSIONS

Ecu = car.CarParams.Ecu
ECU_NAME = {v: k for k, v in Ecu.schema.enumerants.items()}


class TestFordFingerprint(unittest.TestCase):
  def test_fuzzy_ecus_available(self):
    # Asserts ECU keys essential for fuzzy fingerprinting are available on all platforms
    for car_model, ecus in FW_VERSIONS.items():
      with self.subTest(car_model=car_model):
        for fuzzy_ecu in FW_QUERY_CONFIG.fuzzy_ecus:
          self.assertIn(fuzzy_ecu, [e[0] for e in ecus])

  def test_fuzzy_platform_codes(self):
    codes = FW_QUERY_CONFIG.fuzzy_get_platform_codes([
      b'L1MC-14D003-AJ\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'L1MC-14D003-AK\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'L1MC-14D003-AL\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'M1MC-14D003-AB\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ])
    self.assertEqual(codes, {b'1MC-AE', b'1MC-AH', b'1MC-AC', b'1MC-AK',
                             b'1MC-AG', b'1MC-AL', b'1MC-AD', b'1MC-AI',
                             b'1MC-AJ', b'1MC-AF', b'1MC-AB'})

  # def test_test_test(self):
  #   for car_model, fw_by_addr in FW_VERSIONS.items():
  #     for addr, fws in fw_by_addr.items():
  #       if addr[0] in FW_QUERY_CONFIG.fuzzy_ecus:
  #         for f in fws:
  #           platform_codes = FW_QUERY_CONFIG.fuzzy_get_platform_codes([f])
  #           print(car_model, f, platform_codes)
  #           self.assertEqual(1, len(platform_codes))

  # def test_excluded_platforms(self):
  #   # Asserts a list of platforms that will not fuzzy fingerprint due to shared platform codes
  #   # This list can be shrunk as we combine platforms and detect features
  #   excluded_platforms = [
  #     CAR.HYUNDAI_GENESIS,
  #     CAR.IONIQ,
  #     CAR.IONIQ_PHEV_2019,
  #     CAR.IONIQ_PHEV,
  #     CAR.IONIQ_EV_2020,
  #     CAR.IONIQ_EV_LTD,
  #     CAR.IONIQ_HEV_2022,
  #     CAR.SANTA_FE,
  #     CAR.SANTA_FE_2022,
  #     CAR.KIA_STINGER,
  #     CAR.KIA_STINGER_2022,
  #     CAR.GENESIS_G70,
  #     CAR.GENESIS_G70_2020,
  #     CAR.TUCSON_4TH_GEN,
  #     CAR.TUCSON_HYBRID_4TH_GEN,
  #     CAR.KIA_SPORTAGE_HYBRID_5TH_GEN,
  #     CAR.SANTA_CRUZ_1ST_GEN,
  #     CAR.KIA_SPORTAGE_5TH_GEN,
  #   ]
  #
  #   all_platform_codes = defaultdict(set)
  #   for platform, fw_by_addr in FW_VERSIONS.items():
  #     for addr, fws in fw_by_addr.items():
  #       if addr[0] not in FW_QUERY_CONFIG.fuzzy_ecus:
  #         continue
  #
  #       for platform_code in FW_QUERY_CONFIG.fuzzy_get_platform_codes(fws):
  #         all_platform_codes[(addr[1], addr[2], platform_code)].add(platform)
  #
  #   platforms_with_shared_codes = []
  #   for platform, fw_by_addr in FW_VERSIONS.items():
  #     shared_codes = []
  #     for addr, fws in fw_by_addr.items():
  #       if addr[0] not in FW_QUERY_CONFIG.fuzzy_ecus:
  #         continue
  #
  #       for platform_code in FW_QUERY_CONFIG.fuzzy_get_platform_codes(fws):
  #         shared_codes.append(len(all_platform_codes[(addr[1], addr[2], platform_code)]) > 1)
  #
  #     # If all the platform codes for this platform are shared with another platform,
  #     # we cannot fuzzy fingerprint this platform
  #     if all(shared_codes):
  #       platforms_with_shared_codes.append(platform)
  #
  #   self.assertEqual(set(platforms_with_shared_codes), set(excluded_platforms))


if __name__ == "__main__":
  unittest.main()
