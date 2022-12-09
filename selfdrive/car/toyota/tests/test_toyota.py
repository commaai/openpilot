#!/usr/bin/env python3
from parameterized import parameterized
import unittest

from cereal import car
from selfdrive.car.toyota.values import FW_VERSIONS, TSS2_CAR
from selfdrive.car.toyota.interface import CarInterface


class TestToyotaInterfaces(unittest.TestCase):
  @parameterized.expand([(car,) for car in FW_VERSIONS.keys()])
  def test_toyota_interfaces(self, car_name):
    car_params = CarInterface.get_params(car_name)
    assert car_params

    if car_params.steerControlType == car.CarParams.SteerControlType.angle:
      self.assertIn(car_name, TSS2_CAR)


if __name__ == "__main__":
  unittest.main()
