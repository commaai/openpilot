#!/usr/bin/env python3
from hypothesis import given, settings, strategies as st
import unittest

from cereal import car
from selfdrive.car.fw_versions import build_fw_dict
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

  def test_fw_pattern_new(self):
    """Asserts all ECUs can be parsed"""
    for car_model, ecus in FW_VERSIONS.items():
      for ecu, fws in ecus.items():
        for fw in fws:

          print('\ninput', car_model, fw)
          ret = get_platform_codes([fw])
          self.assertTrue(len(ret))
          print('ret', ret)

  # Tests for platform codes, part numbers, and FW dates which Hyundai will use to fuzzy
  # fingerprint in the absence of full FW matches:
  # def test_platform_code_ecus_available(self):
  #   # TODO: add queries for these non-CAN FD cars to get EPS
  #   no_eps_platforms = CANFD_CAR | {CAR.KIA_SORENTO, CAR.KIA_OPTIMA_G4, CAR.KIA_OPTIMA_G4_FL,
  #                                   CAR.SONATA_LF, CAR.TUCSON, CAR.GENESIS_G90, CAR.GENESIS_G80}
  #
  #   # Asserts ECU keys essential for fuzzy fingerprinting are available on all platforms
  #   for car_model, ecus in FW_VERSIONS.items():
  #     with self.subTest(car_model=car_model):
  #       for platform_code_ecu in PLATFORM_CODE_ECUS:
  #         if platform_code_ecu in (Ecu.fwdRadar, Ecu.eps) and car_model == CAR.HYUNDAI_GENESIS:
  #           continue
  #         if platform_code_ecu == Ecu.eps and car_model in no_eps_platforms:
  #           continue
  #         self.assertIn(platform_code_ecu, [e[0] for e in ecus])
  #
  # def test_fw_format(self):
  #   # Asserts:
  #   # - every supported ECU FW version returns one platform code
  #   # - every supported ECU FW version has a part number
  #   # - expected parsing of ECU FW dates
  #
  #   for car_model, ecus in FW_VERSIONS.items():
  #     with self.subTest(car_model=car_model):
  #       for ecu, fws in ecus.items():
  #         if ecu[0] not in PLATFORM_CODE_ECUS:
  #           continue
  #
  #         codes = set()
  #         for fw in fws:
  #           result = get_platform_codes([fw])
  #           self.assertEqual(1, len(result), f"Unable to parse FW: {fw}")
  #           codes |= result
  #
  #         if ecu[0] not in DATE_FW_ECUS or car_model in NO_DATES_PLATFORMS:
  #           self.assertTrue(all({date is None for _, date in codes}))
  #         else:
  #           self.assertTrue(all({date is not None for _, date in codes}))
  #
  #         if car_model == CAR.HYUNDAI_GENESIS:
  #           raise unittest.SkipTest("No part numbers for car model")
  #
  #         # Hyundai places the ECU part number in their FW versions, assert all parsable
  #         # Some examples of valid formats: b"56310-L0010", b"56310L0010", b"56310/M6300"
  #         self.assertTrue(all({b"-" in code for code, _ in codes}),
  #                         f"FW does not have part number: {fw}")

  def test_platform_codes_spot_check(self):
    # Asserts basic platform code parsing behavior for a few cases
    results = get_platform_codes([b"F152607140\x00\x00\x00\x00\x00\x00"])
    self.assertEqual(results, {(b"F1526-07-1", b"40")})

    results = get_platform_codes([b"\x028646F4104100\x00\x00\x00\x008646G5301200\x00\x00\x00\x00"])
    self.assertEqual(results, {(b"8646F-41-04", b"100")})

    # Short version has no part number
    results = get_platform_codes([b"\x0235879000\x00\x00\x00\x00\x00\x00\x00\x00A4701000\x00\x00\x00\x00\x00\x00\x00\x00"])
    self.assertEqual(results, {(b"35-87", b"9000")})

    results = get_platform_codes([
      b"F152607140\x00\x00\x00\x00\x00\x00",
      b"\x028646F4104100\x00\x00\x00\x008646G5301200\x00\x00\x00\x00",
      b"\x0235879000\x00\x00\x00\x00\x00\x00\x00\x00A4701000\x00\x00\x00\x00\x00\x00\x00\x00",
    ])
    self.assertEqual(results, {(b"F1526-07-1", b"40"), (b"8646F-41-04", b"100"), (b"35-87", b"9000")})

  def test_fuzzy_excluded_platforms(self):
    # Asserts a list of platforms that will not fuzzy fingerprint with platform codes due to them being shared.
    # This list can be shrunk as we combine platforms, detect features, and add the hybrid ECU
    excluded_platforms = {
      # # CAR.LEXUS_ESH_TSS2,
      # # CAR.RAV4_TSS2_2022,
      CAR.LEXUS_ES_TSS2,
      # # CAR.RAV4_TSS2,
      # # CAR.RAV4_TSS2_2023,
      # CAR.RAV4_TSS2,
      # # CAR.CAMRY,
      # CAR.HIGHLANDER_TSS2,
      # CAR.RAV4H_TSS2,
      CAR.LEXUS_RX_TSS2,
      # # CAR.CAMRYH_TSS2,
      # CAR.CHR,
      # # CAR.RAV4H,
      # CAR.RAV4H_TSS2_2022,
      # # CAR.HIGHLANDERH_TSS2,
      # # CAR.RAV4,
      # # CAR.CHR_TSS2,
      # CAR.CHRH,
      # CAR.RAV4H_TSS2_2023,
      # CAR.CAMRY_TSS2,
      # CAR.COROLLA_TSS2,
    }

    platforms_with_shared_codes = set()
    for platform, fw_by_addr in FW_VERSIONS.items():
      # if platform != CAR.RAV4_TSS2:
      #   continue
      # if 'RAV4 HYBRID' not in platform:
      #   continue
      print('platform', platform)
      car_fw = []
      for ecu, fw_versions in fw_by_addr.items():
        ecu_name, addr, sub_addr = ecu
        for fw in fw_versions:
          car_fw.append({"ecu": ecu_name, "fwVersion": fw, "address": addr,
                         "subAddress": 0 if sub_addr is None else sub_addr})

      CP = car.CarParams.new_message(carFw=car_fw)
      matches = FW_QUERY_CONFIG.match_fw_to_car_fuzzy(build_fw_dict(CP.carFw))
      print('matches', matches)
      if len(matches) == 1:
        self.assertEqual(list(matches)[0], platform)
      else:
        platforms_with_shared_codes.add(platform)

    self.assertEqual(platforms_with_shared_codes, excluded_platforms, (len(platforms_with_shared_codes), len(FW_VERSIONS)))


if __name__ == "__main__":
  unittest.main()
