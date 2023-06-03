#!/usr/bin/env python3
from collections import defaultdict
import unittest

from cereal import car
from selfdrive.car.hyundai.values import CAR, CANFD_CAR, FW_QUERY_CONFIG, FW_VERSIONS, CAN_GEARS, LEGACY_SAFETY_MODE_CAR, CHECKSUM, CAMERA_SCC_CAR

Ecu = car.CarParams.Ecu
ECU_NAME = {v: k for k, v in Ecu.schema.enumerants.items()}


class TestHyundaiFingerprint(unittest.TestCase):
  def test_canfd_not_in_can_features(self):
    can_specific_feature_list = set.union(*CAN_GEARS.values(), *CHECKSUM.values(), LEGACY_SAFETY_MODE_CAR, CAMERA_SCC_CAR)
    for car_model in CANFD_CAR:
      self.assertNotIn(car_model, can_specific_feature_list, "CAN FD car unexpectedly found in a CAN feature list")

  def test_auxiliary_request_ecu_whitelist(self):
    # Asserts only auxiliary Ecus can exist in database for CAN-FD cars
    whitelisted_ecus = {ecu for r in FW_QUERY_CONFIG.requests for ecu in r.whitelist_ecus if r.auxiliary}

    for car_model in CANFD_CAR:
      ecus = {fw[0] for fw in FW_VERSIONS[car_model].keys()}
      ecus_not_in_whitelist = ecus - whitelisted_ecus
      ecu_strings = ", ".join([f'Ecu.{ECU_NAME[ecu]}' for ecu in ecus_not_in_whitelist])
      self.assertEqual(len(ecus_not_in_whitelist), 0, f'{car_model}: Car model has ECUs not in auxiliary request whitelists: {ecu_strings}')

  def test_certain_ecus_available(self):
    # Asserts certain ecu keys essential for fuzzy fingerprinting are available on all platforms
    for car, ecus in FW_VERSIONS.items():
      for essential_ecu in FW_QUERY_CONFIG.fuzzy_ecus:
        with self.subTest(car=car):
          if car == CAR.HYUNDAI_GENESIS:
            raise unittest.SkipTest
          self.assertIn(essential_ecu, [e[0] for e in ecus])

  def test_fuzzy_platform_codes(self):
    codes = FW_QUERY_CONFIG.fuzzy_get_platform_codes([b'\xf1\x00DH LKAS 1.1 -150210'])
    self.assertEqual(codes, {b"DH"})

    codes = FW_QUERY_CONFIG.fuzzy_get_platform_codes([b'\xf1\x00AEhe SCC H-CUP      1.01 1.01 96400-G2000         '])
    self.assertEqual(codes, {b"AEhe"})

    codes = FW_QUERY_CONFIG.fuzzy_get_platform_codes([b'\xf1\x00CV1_ RDR -----      1.00 1.01 99110-CV000         '])
    self.assertEqual(codes, {b"CV1"})

    codes = FW_QUERY_CONFIG.fuzzy_get_platform_codes([
      b'\xf1\x00DH LKAS 1.1 -150210',
      b'\xf1\x00AEhe SCC H-CUP      1.01 1.01 96400-G2000         ',
      b'\xf1\x00CV1_ RDR -----      1.00 1.01 99110-CV000         ',
    ])
    self.assertEqual(codes, {b"DH", b"AEhe", b"CV1"})

  def test_excluded_platforms(self):
    # Asserts a list of platforms that will not fuzzy fingerprint due to shared platform codes
    # This list can be shrunk as we combine platforms and detect features
    excluded_platforms = [
      CAR.HYUNDAI_GENESIS,
      CAR.IONIQ,
      CAR.IONIQ_PHEV_2019,
      CAR.IONIQ_PHEV,
      CAR.IONIQ_EV_2020,
      CAR.IONIQ_EV_LTD,
      CAR.IONIQ_HEV_2022,
      CAR.SANTA_FE,
      CAR.SANTA_FE_2022,
      CAR.KIA_STINGER,
      CAR.KIA_STINGER_2022,
      CAR.GENESIS_G70,
      CAR.GENESIS_G70_2020,
      CAR.TUCSON_4TH_GEN,
      CAR.TUCSON_HYBRID_4TH_GEN,
      CAR.KIA_SPORTAGE_HYBRID_5TH_GEN,
      CAR.SANTA_CRUZ_1ST_GEN,
      CAR.KIA_SPORTAGE_5TH_GEN,
    ]

    # platform_codes = defaultdict(lambda: defaultdict(set))
    platform_codes = defaultdict(dict)
    all_platform_codes = defaultdict(set)
    for candidate, fw_by_addr in FW_VERSIONS.items():
      for addr, fws in fw_by_addr.items():
        if addr[0] not in FW_QUERY_CONFIG.fuzzy_ecus:
          continue

        for platform_code in FW_QUERY_CONFIG.fuzzy_get_platform_codes(fws):
          all_platform_codes[(addr[1], addr[2], platform_code)].add(candidate)
        # for f in fws:
        #   # Add platform codes to lookup dict if config specifies a function
        #   platform_codes = FW_QUERY_CONFIG.fuzzy_get_platform_codes([f])
        #   # if len(platform_codes) == 1:
        #   all_platform_codes[(addr[1], addr[2], list(platform_codes)[0])].add(candidate)

    # print(all_platform_codes)
    # raise Exception
    platforms_with_shared_codes = []
    for platform, ecus in FW_VERSIONS.items():
      shared_codes = []
      for addr, fws in ecus.items():
        if addr[0] not in FW_QUERY_CONFIG.fuzzy_ecus:
          continue

        for f in fws:
          # if len(platform_codes) == 1:
          platform_code = list(FW_QUERY_CONFIG.fuzzy_get_platform_codes([f]))[0]
          shared_codes.append(len(all_platform_codes[(addr[1], addr[2], platform_code)]) > 1)
      print(platform, shared_codes, not all(shared_codes))
      if all(shared_codes):
        platforms_with_shared_codes.append(platform)
    print(platforms_with_shared_codes)
    # make sure list is unique
    self.assertEqual(len(platforms_with_shared_codes), len(set(platforms_with_shared_codes)))
    self.assertEqual(set(platforms_with_shared_codes), set(excluded_platforms))
    #   if platform != CAR.HYUNDAI_GENESIS:
    #     platform_codes[Ecu.fwdRadar][platform] = FW_QUERY_CONFIG.fuzzy_get_platform_codes(ecus[(Ecu.fwdRadar, 0x7d0, None)])
    #   platform_codes[Ecu.fwdCamera][platform] = FW_QUERY_CONFIG.fuzzy_get_platform_codes(ecus[(Ecu.fwdCamera, 0x7c4, None)])
    #
    # for ecu in [Ecu.fwdRadar, Ecu.fwdCamera]:  # TODO: use fuzzy_ecus
    #   for platform in platform_codes.keys():
    #     pass
    #     # print(platform_codes.values())
    #   print(platform_codes[ecu].values())
    #   print(set.union(*platform_codes[ecu].values()))
    #   print()
    # # for platform in platform_codes.keys():
    # #   print(platform_codes.keys())
    #
    # # print(platform_codes.values())


if __name__ == "__main__":
  unittest.main()
