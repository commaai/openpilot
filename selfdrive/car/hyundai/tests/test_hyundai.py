#!/usr/bin/env python3
import unittest

from cereal import car
from selfdrive.car.hyundai.values import CANFD_CAR, FW_VERSIONS, FW_QUERY_CONFIG

Ecu = car.CarParams.Ecu
ECU_NAME = {v: k for k, v in Ecu.schema.enumerants.items()}


class TestHyundaiFingerprint(unittest.TestCase):
  def test_auxiliary_request_ecu_whitelist(self):
    # Asserts only auxiliary Ecus can exist in database for CAN-FD cars
    whitelisted_ecus = {ecu for r in FW_QUERY_CONFIG.requests for ecu in r.whitelist_ecus if r.bus > 3}

    for car_model in CANFD_CAR:
      ecus = {fw[0] for fw in FW_VERSIONS[car_model].keys()}
      ecus_not_in_whitelist = ecus - whitelisted_ecus
      ecu_strings = ", ".join([f'Ecu.{ECU_NAME[ecu]}' for ecu in ecus_not_in_whitelist])
      self.assertEqual(len(ecus_not_in_whitelist), 0, f'{car_model}: Car model has ECUs not in auxiliary request whitelists: {ecu_strings}')

  def test_certain_ecus_available(self):
    # Asserts certain ecu keys essential for fuzzy fingerprinting are available on all platforms
    essential_ecus = [(Ecu.fwdCamera, 0x7c4, None), (Ecu.fwdRadar, 0x7d0, None)]

    for car, ecus in FW_VERSIONS.items():
      for essential_ecu in essential_ecus:
        self.assertIn(essential_ecu, ecus)


if __name__ == "__main__":
  unittest.main()
