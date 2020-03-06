#!/usr/bin/env python3
import unittest
import importlib
from selfdrive.car.fingerprints import all_known_cars
from selfdrive.car.car_helpers import interfaces
from selfdrive.car.fingerprints import _FINGERPRINTS as FINGERPRINTS


class TestCarParam(unittest.TestCase):
  def test_creating_car_params(self):
    all_cars = all_known_cars()

    for car in all_cars:
      fingerprint = FINGERPRINTS[car][0]

      CarInterface, CarController, CarState = interfaces[car]
      fingerprints = {
        0: fingerprint,
        1: fingerprint,
        2: fingerprint,
      }

      car_fw = []

      for has_relay in [True, False]:
        car_params = CarInterface.get_params(car, fingerprints, has_relay, car_fw)
        car_interface = CarInterface(car_params, CarController, CarState), car_params

        RadarInterface = importlib.import_module('selfdrive.car.%s.radar_interface' % car_params.carName).RadarInterface
        radar_interface = RadarInterface(car_params)

        assert car_params
        assert car_interface
        assert radar_interface


if __name__ == "__main__":
  unittest.main()
