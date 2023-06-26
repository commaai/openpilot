#!/usr/bin/env python3
import unittest

from selfdrive.car.toyota.values import TSS2_CAR, ANGLE_CONTROL_CAR


class TestToyotaInterfaces(unittest.TestCase):
  def test_angle_car_set(self):
    self.assertTrue(len(ANGLE_CONTROL_CAR - TSS2_CAR) == 0)


if __name__ == "__main__":
  unittest.main()
