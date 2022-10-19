#!/usr/bin/env python3
import unittest
from parameterized import parameterized

from cereal import car
from selfdrive.car.car_helpers import get_interface_attr, interfaces
from selfdrive.car.fw_versions import FW_QUERY_CONFIGS
from selfdrive.car.hyundai.values import CANFD_CAR, FW_VERSIONS, FINGERPRINTS

Ecu = car.CarParams.Ecu

ECU_NAME = {v: k for k, v in Ecu.schema.enumerants.items()}
VERSIONS = get_interface_attr("FW_VERSIONS", ignore_none=True)
ALL_HYUNDAI_CARS = list({*FW_VERSIONS.keys(), *FINGERPRINTS.keys()})


class TestHyundaiFingerprint(unittest.TestCase):
  def test_auxiliary_request_ecu_whitelist(self):
    # Asserts only auxiliary Ecus can exist in database for CAN-FD cars
    config = FW_QUERY_CONFIGS['hyundai']
    whitelisted_ecus = {ecu for r in config.requests for ecu in r.whitelist_ecus if r.bus > 3}

    for car_model in CANFD_CAR:
      ecus = {fw[0] for fw in VERSIONS['hyundai'][car_model].keys()}
      ecus_not_in_whitelist = ecus - whitelisted_ecus
      ecu_strings = ", ".join([f'Ecu.{ECU_NAME[ecu]}' for ecu in ecus_not_in_whitelist])
      self.assertEqual(len(ecus_not_in_whitelist), 0, f'{car_model}: Car model has ECUs not in auxiliary request whitelists: {ecu_strings}')


class TestHyundaiInterface(unittest.TestCase):
  @parameterized.expand([(car,) for car in ALL_HYUNDAI_CARS])
  def test_car_control_params(self, car_model):
    CP = interfaces[car_model][0].get_params(car_model)
    self.assertGreater(CP.carControlParams.steerMax, 0, f"{car_model}: steerMax not set")

    divisible = int(CP.carControlParams.steerMax / 10) * 10
    self.assertEqual(CP.carControlParams.steerMax, divisible, f"{car_model}: steerMax not divisible by 10")


if __name__ == "__main__":
  unittest.main()
