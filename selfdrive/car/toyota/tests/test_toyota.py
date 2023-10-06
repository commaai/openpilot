#!/usr/bin/env python3
from hypothesis import given, settings, strategies as st
import unittest

from cereal import car
from openpilot.selfdrive.car.toyota.values import CAR, DBC, TSS2_CAR, ANGLE_CONTROL_CAR, RADAR_ACC_CAR, FW_VERSIONS, \
                                                  PLATFORM_CODE_ECUS, get_platform_codes

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
      with self.subTest(car_model=car_model.value):
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
  # Tests for part numbers, platform codes, and sub-versions which Toyota will use to fuzzy
  # fingerprint in the absence of full FW matches:
  @settings(max_examples=100)
  @given(data=st.data())
  def test_platform_codes_fuzzy_fw(self, data):
    fw_strategy = st.lists(st.binary())
    fws = data.draw(fw_strategy)
    get_platform_codes(fws)

  def test_platform_code_ecus_available(self):
    # Asserts ECU keys essential for fuzzy fingerprinting are available on all platforms
    for car_model, ecus in FW_VERSIONS.items():
      with self.subTest(car_model=car_model.value):
        for platform_code_ecu in PLATFORM_CODE_ECUS:
          if platform_code_ecu == Ecu.eps and car_model in (CAR.PRIUS_V, CAR.LEXUS_CTH,):
            continue
          if platform_code_ecu == Ecu.abs and car_model in (CAR.ALPHARD_TSS2,):
            continue
          self.assertIn(platform_code_ecu, [e[0] for e in ecus])

  def test_fw_format(self):
    # Asserts:
    # - every supported ECU FW version returns one platform code
    # - every supported ECU FW version has a part number
    # - expected parsing of ECU sub-versions

    for car_model, ecus in FW_VERSIONS.items():
      with self.subTest(car_model=car_model.value):
        for ecu, fws in ecus.items():
          if ecu[0] not in PLATFORM_CODE_ECUS:
            continue

          codes = dict()
          for fw in fws:
            result = get_platform_codes([fw])
            # Check only one platform code and sub-version
            self.assertEqual(1, len(result), f"Unable to parse FW: {fw}")
            self.assertEqual(1, len(list(result.values())[0]), f"Unable to parse FW: {fw}")
            codes |= result

          # Toyota places the ECU part number in their FW versions, assert all parsable
          # Note that there is only one unique part number per ECU across the fleet, so this
          # is not important for identification, just a sanity check.
          self.assertTrue(all(code.count(b"-") > 1 for code in codes),
                          f"FW does not have part number: {fw} {codes}")

  def test_platform_codes_spot_check(self):
    # Asserts basic platform code parsing behavior for a few cases
    results = get_platform_codes([
      b"F152607140\x00\x00\x00\x00\x00\x00",
      b"F152607171\x00\x00\x00\x00\x00\x00",
      b"F152607110\x00\x00\x00\x00\x00\x00",
      b"F152607180\x00\x00\x00\x00\x00\x00",
    ])
    self.assertEqual(results, {b"F1526-07-1": {b"10", b"40", b"71", b"80"}})

    results = get_platform_codes([
      b"\x028646F4104100\x00\x00\x00\x008646G5301200\x00\x00\x00\x00",
      b"\x028646F4104100\x00\x00\x00\x008646G3304000\x00\x00\x00\x00",
    ])
    self.assertEqual(results, {b"8646F-41-04": {b"100"}})

    # Short version has no part number
    results = get_platform_codes([
      b"\x0235870000\x00\x00\x00\x00\x00\x00\x00\x00A0202000\x00\x00\x00\x00\x00\x00\x00\x00",
      b"\x0235883000\x00\x00\x00\x00\x00\x00\x00\x00A0202000\x00\x00\x00\x00\x00\x00\x00\x00",
    ])
    self.assertEqual(results, {b"58-70": {b"000"}, b"58-83": {b"000"}})

    results = get_platform_codes([
      b"F152607110\x00\x00\x00\x00\x00\x00",
      b"F152607140\x00\x00\x00\x00\x00\x00",
      b"\x028646F4104100\x00\x00\x00\x008646G5301200\x00\x00\x00\x00",
      b"\x0235879000\x00\x00\x00\x00\x00\x00\x00\x00A4701000\x00\x00\x00\x00\x00\x00\x00\x00",
    ])
    self.assertEqual(results, {b"F1526-07-1": {b"10", b"40"}, b"8646F-41-04": {b"100"}, b"58-79": {b"000"}})


if __name__ == "__main__":
  unittest.main()
