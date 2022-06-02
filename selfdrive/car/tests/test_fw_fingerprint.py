#!/usr/bin/env python3
import random
import unittest
from parameterized import parameterized

from cereal import car
from selfdrive.car.car_helpers import get_interface_attr, interfaces
from selfdrive.car.fingerprints import FW_VERSIONS
from selfdrive.car.fw_versions import REQUESTS, match_fw_to_car

CarFw = car.CarParams.CarFw
Ecu = car.CarParams.Ecu

ECU_NAME = {v: k for k, v in Ecu.schema.enumerants.items()}


class TestFwFingerprint(unittest.TestCase):
  def assertFingerprints(self, candidates, expected):
    candidates = list(candidates)
    self.assertEqual(len(candidates), 1, f"got more than one candidate: {candidates}")
    self.assertEqual(candidates[0], expected)

  @parameterized.expand([(k, v) for k, v in FW_VERSIONS.items()])
  def test_fw_fingerprint(self, car_model, ecus):
    CP = car.CarParams.new_message()
    for _ in range(200):
      fw = []
      for ecu, fw_versions in ecus.items():
        ecu_name, addr, sub_addr = ecu
        fw.append({"ecu": ecu_name, "fwVersion": random.choice(fw_versions),
                   "address": addr, "subAddress": 0 if sub_addr is None else sub_addr})
      CP.carFw = fw
      _, matches = match_fw_to_car(CP.carFw)
      self.assertFingerprints(matches, car_model)

  def test_no_duplicate_fw_versions(self):
    passed = True
    for car_model, ecus in FW_VERSIONS.items():
      for ecu, ecu_fw in ecus.items():
        duplicates = {fw for fw in ecu_fw if ecu_fw.count(fw) > 1}
        if len(duplicates):
          print(car_model, ECU_NAME[ecu[0]], duplicates)
          passed = False

    self.assertTrue(passed, "Duplicate FW versions found")

  def test_blacklisted_ecus(self):
    passed = True
    blacklisted_addrs = (0x7c4, 0x7d0)  # includes A/C ecu and an unknown ecu
    for car_model, ecus in FW_VERSIONS.items():
      CP = interfaces[car_model][0].get_params(car_model)
      if CP.carName == 'subaru':
        for ecu in ecus.keys():
          if ecu[1] in blacklisted_addrs:
            print(f'{car_model}: Blacklisted ecu: (Ecu.{ECU_NAME[ecu[0]]}, {hex(ecu[1])})')
            passed = False

    self.assertTrue(passed, "Blacklisted FW versions found")

  def test_fw_request_ecu_whitelist(self):
    passed = True
    brands = set(r.brand for r in REQUESTS)
    versions = get_interface_attr('FW_VERSIONS')
    for brand in brands:
      whitelist_ecus = [ecu for r in REQUESTS for ecu in r.whitelist_ecus if r.brand == brand]
      if len(whitelist_ecus) == 0:
        continue  # no whitelists for brand

      # each ecu in fw versions needs to be whitelisted at least once
      brand_ecu_types = set()
      for fw_versions in versions[brand].values():
        for (ecu_type, _, _) in fw_versions.keys():
          brand_ecu_types.add(ecu_type)

      ecus_not_in_whitelists = set(brand_ecu_types) - set(whitelist_ecus)
      if len(ecus_not_in_whitelists):
        ecu_strings = ", ".join([f'Ecu.{ECU_NAME[ecu]}' for ecu in ecus_not_in_whitelists])
        print(f'{brand.title()} FW request does not whitelist ecus: {ecu_strings}')
        passed = False

    self.assertTrue(passed, "Not all ecus in FW versions found in request whitelists")


if __name__ == "__main__":
  unittest.main()
