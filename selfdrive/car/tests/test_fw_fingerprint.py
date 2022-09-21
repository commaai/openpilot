#!/usr/bin/env python3
import random
import unittest
from parameterized import parameterized

from cereal import car
from selfdrive.car.car_helpers import get_interface_attr, interfaces
from selfdrive.car.fingerprints import FW_VERSIONS
from selfdrive.car.fw_versions import FW_QUERY_CONFIGS, match_fw_to_car

CarFw = car.CarParams.CarFw
Ecu = car.CarParams.Ecu

ECU_NAME = {v: k for k, v in Ecu.schema.enumerants.items()}
VERSIONS = get_interface_attr("FW_VERSIONS", ignore_none=True)


def get_brand_ecus(brand):
  return {fw for car_fw in VERSIONS[brand].values() for fw in car_fw}


class TestFwFingerprint(unittest.TestCase):
  def assertFingerprints(self, candidates, expected):
    candidates = list(candidates)
    self.assertEqual(len(candidates), 1, f"got more than one candidate: {candidates}")
    self.assertEqual(candidates[0], expected)

  @parameterized.expand([(b, c, e[c]) for b, e in VERSIONS.items() for c in e])
  def test_fw_fingerprint(self, brand, car_model, ecus):
    CP = car.CarParams.new_message()
    for _ in range(200):
      fw = []
      for ecu, fw_versions in ecus.items():
        if not len(fw_versions):
          raise unittest.SkipTest("Car model has no FW versions")
        ecu_name, addr, sub_addr = ecu
        fw.append({"ecu": ecu_name, "fwVersion": random.choice(fw_versions), 'brand': brand,
                   "address": addr, "subAddress": 0 if sub_addr is None else sub_addr})
      CP.carFw = fw
      _, matches = match_fw_to_car(CP.carFw)
      self.assertFingerprints(matches, car_model)

  def test_no_duplicate_fw_versions(self):
    for car_model, ecus in FW_VERSIONS.items():
      with self.subTest(car_model=car_model):
        for ecu, ecu_fw in ecus.items():
          with self.subTest(ecu):
            duplicates = {fw for fw in ecu_fw if ecu_fw.count(fw) > 1}
            self.assertFalse(len(duplicates), f"{car_model}: Duplicate FW versions: Ecu.{ECU_NAME[ecu[0]]}, {duplicates}")

  def test_blacklisted_ecus(self):
    blacklisted_addrs = (0x7c4, 0x7d0)  # includes A/C ecu and an unknown ecu
    for car_model, ecus in FW_VERSIONS.items():
      with self.subTest(car_model=car_model):
        CP = interfaces[car_model][0].get_params(car_model)
        if CP.carName == 'subaru':
          for ecu in ecus.keys():
            self.assertNotIn(ecu[1], blacklisted_addrs, f'{car_model}: Blacklisted ecu: (Ecu.{ECU_NAME[ecu[0]]}, {hex(ecu[1])})')

        elif CP.carName == "chrysler":
          # Some HD trucks have a combined TCM and ECM
          if CP.carFingerprint.startswith("RAM HD"):
            for ecu in ecus.keys():
              self.assertNotEqual(ecu[0], Ecu.transmission, f"{car_model}: Blacklisted ecu: (Ecu.{ECU_NAME[ecu[0]]}, {hex(ecu[1])})")

  def test_missing_versions_and_configs(self):
    brand_versions = set(VERSIONS.keys())
    brand_configs = set(FW_QUERY_CONFIGS.keys())
    if len(brand_configs - brand_versions):
      with self.subTest():
        self.fail(f"Brands do not implement FW_VERSIONS: {brand_configs - brand_versions}")

    if len(brand_versions - brand_configs):
      with self.subTest():
        self.fail(f"Brands do not implement FW_QUERY_CONFIG: {brand_versions - brand_configs}")

  def test_config_ecus(self):
    for brand, config in FW_QUERY_CONFIGS.items():
      with self.subTest(brand=brand):
        brand_ecus = {ecu for ecu, _, _ in get_brand_ecus(brand)}
        brand_addrs = {(addr, sub_addr) for _, addr, sub_addr in get_brand_ecus(brand)}

        # Tests all defined ecus are able to be fingerprinted with
        for (addr, sub_addr), ecu in config.ecus.items():
          ecu_repr = f"(Ecu.{ECU_NAME[ecu]}, {hex(addr)})"
          self.assertIn(ecu, brand_ecus, f"{brand}: Ecu specified in FwQueryConfig does not exist in versions: {ecu_repr}")
          self.assertIn((addr, sub_addr), brand_addrs, f"{brand}: Ecu specified in FwQueryConfig does not exist in versions: {ecu_repr}")

        # Tests all extra ecus are purely added for querying and data collection; not to be fingerprinted on
        for (addr, sub_addr), ecu in config.extra_ecus.items():
          ecu_repr = f"(Ecu.{ECU_NAME[ecu]}, {hex(addr)})"
          self.assertNotIn((addr, sub_addr), brand_addrs, f"{brand}: Extra ecu specified in FwQueryConfig exists in versions: {ecu_repr}")

  def test_all_version_ecus_exist_in_config(self):
    for brand, config in FW_QUERY_CONFIGS.items():
      with self.subTest(brand=brand):
        for ecu, addr, sub_addr in get_brand_ecus(brand):
          ecu_repr = f"(Ecu.{ECU_NAME[ecu]}, {hex(addr)})"
          self.assertIn(ecu, config.ecus.values(), f"{brand}: Ecu specified in versions does not exist in FwQueryConfig: {ecu_repr}")
          self.assertIn((addr, sub_addr), config.ecus, f"{brand}: Ecu specified in versions does not exist in FwQueryConfig: {ecu_repr}")

  def test_fw_request_ecu_whitelist(self):
    for brand, config in FW_QUERY_CONFIGS.items():
      with self.subTest(brand=brand):
        whitelisted_ecus = set([ecu for r in config.requests for ecu in r.whitelist_ecus])
        brand_ecus = {ecu for ecu, _, _ in get_brand_ecus(brand)}

        # each ecu in brand's fw versions needs to be whitelisted at least once
        ecus_not_whitelisted = brand_ecus - whitelisted_ecus

        ecu_strings = ", ".join([f'Ecu.{ECU_NAME[ecu]}' for ecu in ecus_not_whitelisted])
        self.assertFalse(len(whitelisted_ecus) and len(ecus_not_whitelisted),
                         f'{brand.title()}: FW query whitelist missing ecus: {ecu_strings}')


if __name__ == "__main__":
  unittest.main()
