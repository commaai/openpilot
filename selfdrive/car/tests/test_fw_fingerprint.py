#!/usr/bin/env python3
import random
import unittest
from itertools import product
from parameterized import parameterized

from cereal import car
from selfdrive.car.fingerprints import FW_VERSIONS
from selfdrive.car.fw_versions import match_fw_to_car
from selfdrive.car.toyota.values import CAR as TOYOTA

CarFw = car.CarParams.CarFw
Ecu = car.CarParams.Ecu

ECU_NAME = {v: k for k, v in Ecu.schema.enumerants.items()}

class TestFwFingerprint(unittest.TestCase):
  def assertFingerprints(self, candidates, expected):
    candidates = list(candidates)
    self.assertEqual(len(candidates), 1)
    self.assertEqual(candidates[0], expected)

  def test_rav4_tss2(self):
    CP = car.CarParams.new_message()
    CP.carFw = [
      {"ecu": Ecu.esp,
       "fwVersion": b"\x01F15260R210\x00\x00\x00\x00\x00\x00",
       "address": 1968,
       "subAddress": 0},
      {"ecu": Ecu.engine,
       "fwVersion": b"\x028966342Y8000\x00\x00\x00\x00897CF1201001\x00\x00\x00\x00",
       "address": 1792,
       "subAddress": 0},
      {"ecu": Ecu.eps,
       "fwVersion": b"\x028965B0R01200\x00\x00\x00\x008965B0R02200\x00\x00\x00\x00",
       "address": 1953,
       "subAddress": 0},
      {"ecu": Ecu.fwdRadar,
       "fwVersion": b"\x018821F3301200\x00\x00\x00\x00",
       "address": 1872,
       "subAddress": 15},
      {"ecu": Ecu.fwdCamera,
       "fwVersion": b"\x028646F4203300\x00\x00\x00\x008646G26011A0\x00\x00\x00\x00",
       "address": 1872,
       "subAddress": 109}
    ]

    self.assertFingerprints(match_fw_to_car(CP.carFw), TOYOTA.RAV4_TSS2)

  @parameterized.expand([(k, v) for k, v in FW_VERSIONS.items()])
  def test_fw_fingerprint_all(self, car_model, ecus):
    # TODO: this is too slow, so don't run for now
    return

    ecu_fw_lists = []  # pylint: disable=W0101
    for ecu, fw_versions in ecus.items():
      ecu_name, addr, sub_addr = ecu
      ecu_fw_lists.append([])
      for fw in fw_versions:
        ecu_fw_lists[-1].append({"ecu": ecu_name, "fwVersion": fw, "address": addr,
                                 "subAddress": 0 if sub_addr is None else sub_addr})
    CP = car.CarParams.new_message()
    for car_fw in product(*ecu_fw_lists):
      CP.carFw = car_fw
      self.assertFingerprints(match_fw_to_car(CP.carFw), car_model)

  @parameterized.expand([(k, v) for k, v in FW_VERSIONS.items()])
  def test_fw_fingerprint(self, car_model, ecus):
    CP = car.CarParams.new_message()
    for _ in range(20):
      fw = []
      for ecu, fw_versions in ecus.items():
        ecu_name, addr, sub_addr = ecu
        fw.append({"ecu": ecu_name, "fwVersion": random.choice(fw_versions),
                         "address": addr, "subAddress": 0 if sub_addr is None else sub_addr})
      CP.carFw = fw
      self.assertFingerprints(match_fw_to_car(CP.carFw), car_model)

  def test_no_duplicate_fw_versions(self):
    passed = True
    for car_model, ecus in FW_VERSIONS.items():
      for ecu, ecu_fw in ecus.items():
        duplicates = set([fw for fw in ecu_fw if ecu_fw.count(fw) > 1])
        if len(duplicates):
          print(car_model, ECU_NAME[ecu[0]], duplicates)
          passed = False

    self.assertTrue(passed, "Duplicate FW versions found")

if __name__ == "__main__":
  unittest.main()
