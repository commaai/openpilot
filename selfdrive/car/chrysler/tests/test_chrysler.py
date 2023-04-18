#!/usr/bin/env python3
import unittest

from cereal import car
from selfdrive.car.chrysler.values import FW_VERSIONS

Ecu = car.CarParams.Ecu
ECU_NAME = {v: k for k, v in Ecu.schema.enumerants.items()}


class TestHyundaiFingerprint(unittest.TestCase):
  def test_blacklisted_ecus(self):
    for car_model, ecus in FW_VERSIONS.items():
      with self.subTest(car_model=car_model):
        # Some HD trucks have a combined TCM and ECM
        if car_model.startswith("RAM HD"):
          for ecu in ecus.keys():
            self.assertNotEqual(ecu[0], Ecu.transmission, f"{car_model}: Blacklisted ecu: (Ecu.{ECU_NAME[ecu[0]]}, {hex(ecu[1])})")


if __name__ == "__main__":
  unittest.main()
