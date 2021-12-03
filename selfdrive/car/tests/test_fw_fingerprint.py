#!/usr/bin/env python3
import random
import unittest
from parameterized import parameterized

from cereal import car
from selfdrive.car.fingerprints import FW_VERSIONS
from selfdrive.car.fw_versions import match_fw_to_car

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
        duplicates = set(fw for fw in ecu_fw if ecu_fw.count(fw) > 1)
        if len(duplicates):
          print(car_model, ECU_NAME[ecu[0]], duplicates)
          passed = False

    self.assertTrue(passed, "Duplicate FW versions found")

if __name__ == "__main__":
  unittest.main()
