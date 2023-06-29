#!/usr/bin/env python3
import unittest

from selfdrive.car.toyota.values import DBC, TSS2_CAR, ANGLE_CONTROL_CAR


class TestToyotaInterfaces(unittest.TestCase):
  def test_angle_car_set(self):
    self.assertTrue(len(ANGLE_CONTROL_CAR - TSS2_CAR) == 0)

  def test_tss2_dbc(self):
    # We make some assumptions about TSS2 platforms,
    # like looking up certain signals only in this DBC
    for car, dbc in DBC.items():
      if car in TSS2_CAR:
        self.assertEqual(dbc["pt"], "toyota_nodsu_pt_generated")


if __name__ == "__main__":
  unittest.main()
