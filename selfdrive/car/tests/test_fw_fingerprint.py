#!/usr/bin/env python3
import unittest
from cereal import car
from selfdrive.car.fw_versions import match_fw_to_car
from selfdrive.car.toyota.values import CAR as TOYOTA

CarFw = car.CarParams.CarFw
Ecu = car.CarParams.Ecu


class TestFwFingerprint(unittest.TestCase):
  def assertFingerprints(self, candidates, expected):
    candidates = list(candidates)
    self.assertEqual(len(candidates), 1)
    self.assertEqual(candidates[0], TOYOTA.RAV4_TSS2)

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


if __name__ == "__main__":
  unittest.main()
