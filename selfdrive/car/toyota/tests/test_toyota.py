#!/usr/bin/env python3
from cereal import car
import unittest

from selfdrive.car.toyota.values import TSS2_CAR, ANGLE_CONTROL_CAR, FW_VERSIONS

Ecu = car.CarParams.Ecu
ECU_NAME = {v: k for k, v in Ecu.schema.enumerants.items()}


class TestToyotaInterfaces(unittest.TestCase):
  def test_angle_car_set(self):
    self.assertTrue(len(ANGLE_CONTROL_CAR - TSS2_CAR) == 0)


class TestToyotaFingerprint(unittest.TestCase):
  def test_essential_ecus(self):
    # Asserts ECU keys essential for fuzzy fingerprinting are available on all platforms
    essential_ecus = {Ecu.fwdRadar, Ecu.fwdCamera, Ecu.engine}
    for car_model, ecus in FW_VERSIONS.items():
      with self.subTest(car_model=car_model):
        missing_ecus = essential_ecus - {ecu[0] for ecu in ecus}
        self.assertEqual(missing_ecus, set())
        # for platform_code_ecu in PLATFORM_CODE_ECUS:
        #   if platform_code_ecu in (Ecu.fwdRadar, Ecu.eps) and car_model == CAR.HYUNDAI_GENESIS:
        #     continue
        #   if platform_code_ecu == Ecu.eps and car_model in no_eps_platforms:
        #     continue
        #   self.assertIn(platform_code_ecu, [e[0] for e in ecus])


if __name__ == "__main__":
  unittest.main()
