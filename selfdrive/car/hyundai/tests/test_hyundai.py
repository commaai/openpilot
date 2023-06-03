#!/usr/bin/env python3
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



if __name__ == "__main__":
  unittest.main()
