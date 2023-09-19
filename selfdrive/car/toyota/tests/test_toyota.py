#!/usr/bin/env python3
from hypothesis import given, settings, strategies as st
import unittest

from cereal import car
from openpilot.selfdrive.car.toyota.values import CAR, DBC, TSS2_CAR, ANGLE_CONTROL_CAR, RADAR_ACC_CAR, FW_VERSIONS, \
                                                  get_platform_codes

Ecu = car.CarParams.Ecu
ECU_NAME = {v: k for k, v in Ecu.schema.enumerants.items()}


class TestToyotaInterfaces(unittest.TestCase):
  def test_car_sets(self):
    self.assertTrue(len(ANGLE_CONTROL_CAR - TSS2_CAR) == 0)
    self.assertTrue(len(RADAR_ACC_CAR - TSS2_CAR) == 0)

  def test_tss2_dbc(self):
    # We make some assumptions about TSS2 platforms,
    # like looking up certain signals only in this DBC
    for car_model, dbc in DBC.items():
      if car_model in TSS2_CAR:
        self.assertEqual(dbc["pt"], "toyota_nodsu_pt_generated")

  def test_essential_ecus(self):
    # Asserts standard ECUs exist for each platform
    common_ecus = {Ecu.fwdRadar, Ecu.fwdCamera}
    for car_model, ecus in FW_VERSIONS.items():
      with self.subTest(car_model=car_model):
        present_ecus = {ecu[0] for ecu in ecus}
        missing_ecus = common_ecus - present_ecus
        self.assertEqual(len(missing_ecus), 0)

        # Some exceptions for other common ECUs
        if car_model not in (CAR.ALPHARD_TSS2,):
          self.assertIn(Ecu.abs, present_ecus)

        if car_model not in (CAR.MIRAI,):
          self.assertIn(Ecu.engine, present_ecus)

        if car_model not in (CAR.PRIUS_V, CAR.LEXUS_CTH):
          self.assertIn(Ecu.eps, present_ecus)


class TestToyotaFingerprint(unittest.TestCase):
  @settings(max_examples=100)
  @given(data=st.data())
  def test_platform_codes_fuzzy_fw(self, data):
    fw_strategy = st.lists(st.binary())
    fws = data.draw(fw_strategy)
    get_platform_codes(fws)

  def test_fw_pattern(self):
    """Asserts all ECUs can be parsed"""
    for ecus in FW_VERSIONS.values():
      for fws in ecus.values():
        for fw in fws:
          ret = get_platform_codes([fw])
          self.assertTrue(len(ret))


if __name__ == "__main__":
  unittest.main()
