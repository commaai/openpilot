#!/usr/bin/env python3
from hypothesis import settings, given, strategies as st
import unittest

from cereal import car
from openpilot.selfdrive.car.fw_versions import build_fw_dict
from openpilot.selfdrive.car.hyundai.values import CAMERA_SCC_CAR, CANFD_CAR, CAN_GEARS, CAR, CHECKSUM, DATE_FW_ECUS, \
                                         HYBRID_CAR, EV_CAR, FW_QUERY_CONFIG, LEGACY_SAFETY_MODE_CAR, CANFD_FUZZY_WHITELIST, \
                                         UNSUPPORTED_LONGITUDINAL_CAR, PLATFORM_CODE_ECUS, HYUNDAI_VERSION_REQUEST_LONG, \
                                         get_platform_codes
from openpilot.selfdrive.car.hyundai.fingerprints import FW_VERSIONS

Ecu = car.CarParams.Ecu
ECU_NAME = {v: k for k, v in Ecu.schema.enumerants.items()}

# Some platforms have date codes in a different format we don't yet parse (or are missing).
# For now, assert list of expected missing date cars
NO_DATES_PLATFORMS = {
  # CAN FD
  CAR.KIA_SPORTAGE_5TH_GEN,
  CAR.SANTA_CRUZ_1ST_GEN,
  CAR.TUCSON_4TH_GEN,
  # CAN
  CAR.ELANTRA,
  CAR.ELANTRA_GT_I30,
  CAR.KIA_CEED,
  CAR.KIA_FORTE,
  CAR.KIA_OPTIMA_G4,
  CAR.KIA_OPTIMA_G4_FL,
  CAR.KIA_SORENTO,
  CAR.KONA,
  CAR.KONA_EV,
  CAR.KONA_EV_2022,
  CAR.KONA_HEV,
  CAR.SONATA_LF,
  CAR.VELOSTER,
}


class TestHyundaiFingerprint(unittest.TestCase):
  def test_can_features(self):
    # Test no EV/HEV in any gear lists (should all use ELECT_GEAR)
    self.assertEqual(set.union(*CAN_GEARS.values()) & (HYBRID_CAR | EV_CAR), set())

    # Test CAN FD car not in CAN feature lists
    can_specific_feature_list = set.union(*CAN_GEARS.values(), *CHECKSUM.values(), LEGACY_SAFETY_MODE_CAR, UNSUPPORTED_LONGITUDINAL_CAR, CAMERA_SCC_CAR)
    for car_model in CANFD_CAR:
      self.assertNotIn(car_model, can_specific_feature_list, "CAN FD car unexpectedly found in a CAN feature list")

  def test_hybrid_ev_sets(self):
    self.assertEqual(HYBRID_CAR & EV_CAR, set(), "Shared cars between hybrid and EV")
    self.assertEqual(CANFD_CAR & HYBRID_CAR, set(), "Hard coding CAN FD cars as hybrid is no longer supported")

  def test_auxiliary_request_ecu_whitelist(self):
    # Asserts only auxiliary Ecus can exist in database for CAN-FD cars
    whitelisted_ecus = {ecu for r in FW_QUERY_CONFIG.requests for ecu in r.whitelist_ecus if r.auxiliary}

    for car_model in CANFD_CAR:
      ecus = {fw[0] for fw in FW_VERSIONS[car_model].keys()}
      ecus_not_in_whitelist = ecus - whitelisted_ecus
      ecu_strings = ", ".join([f"Ecu.{ECU_NAME[ecu]}" for ecu in ecus_not_in_whitelist])
      self.assertEqual(len(ecus_not_in_whitelist), 0,
                       f"{car_model}: Car model has ECUs not in auxiliary request whitelists: {ecu_strings}")

  def test_blacklisted_parts(self):
    # Asserts no ECUs known to be shared across platforms exist in the database.
    # Tucson having Santa Cruz camera and EPS for example
    for car_model, ecus in FW_VERSIONS.items():
      with self.subTest(car_model=car_model.value):
        if car_model == CAR.SANTA_CRUZ_1ST_GEN:
          raise unittest.SkipTest("Skip checking Santa Cruz for its parts")

        for code, _ in get_platform_codes(ecus[(Ecu.fwdCamera, 0x7c4, None)]):
          if b"-" not in code:
            continue
          part = code.split(b"-")[1]
          self.assertFalse(part.startswith(b'CW'), "Car has bad part number")

  def test_correct_ecu_response_database(self):
    """
    Assert standard responses for certain ECUs, since they can
    respond to multiple queries with different data
    """
    expected_fw_prefix = HYUNDAI_VERSION_REQUEST_LONG[1:]
    for car_model, ecus in FW_VERSIONS.items():
      with self.subTest(car_model=car_model.value):
        for ecu, fws in ecus.items():
          # TODO: enable for Ecu.fwdRadar, Ecu.abs, Ecu.eps, Ecu.transmission
          if ecu[0] in (Ecu.fwdCamera,):
            self.assertTrue(all(fw.startswith(expected_fw_prefix) for fw in fws),
                            f"FW from unexpected request in database: {(ecu, fws)}")

  @settings(max_examples=100)
  @given(data=st.data())
  def test_platform_codes_fuzzy_fw(self, data):
    """Ensure function doesn't raise an exception"""
    fw_strategy = st.lists(st.binary())
    fws = data.draw(fw_strategy)
    get_platform_codes(fws)

  def test_expected_platform_codes(self):
    # Ensures we don't accidentally add multiple platform codes for a car unless it is intentional
    for car_model, ecus in FW_VERSIONS.items():
      with self.subTest(car_model=car_model.value):
        for ecu, fws in ecus.items():
          if ecu[0] not in PLATFORM_CODE_ECUS:
            continue

          # Third and fourth character are usually EV/hybrid identifiers
          codes = {code.split(b"-")[0][:2] for code, _ in get_platform_codes(fws)}
          if car_model == CAR.PALISADE:
            self.assertEqual(codes, {b"LX", b"ON"}, f"Car has unexpected platform codes: {car_model} {codes}")
          elif car_model == CAR.KONA_EV and ecu[0] == Ecu.fwdCamera:
            self.assertEqual(codes, {b"OE", b"OS"}, f"Car has unexpected platform codes: {car_model} {codes}")
          else:
            self.assertEqual(len(codes), 1, f"Car has multiple platform codes: {car_model} {codes}")

  # Tests for platform codes, part numbers, and FW dates which Hyundai will use to fuzzy
  # fingerprint in the absence of full FW matches:
  def test_platform_code_ecus_available(self):
    # TODO: add queries for these non-CAN FD cars to get EPS
    no_eps_platforms = CANFD_CAR | {CAR.KIA_SORENTO, CAR.KIA_OPTIMA_G4, CAR.KIA_OPTIMA_G4_FL, CAR.KIA_OPTIMA_H,
                                    CAR.KIA_OPTIMA_H_G4_FL, CAR.SONATA_LF, CAR.TUCSON, CAR.GENESIS_G90, CAR.GENESIS_G80, CAR.ELANTRA}

    # Asserts ECU keys essential for fuzzy fingerprinting are available on all platforms
    for car_model, ecus in FW_VERSIONS.items():
      with self.subTest(car_model=car_model.value):
        for platform_code_ecu in PLATFORM_CODE_ECUS:
          if platform_code_ecu in (Ecu.fwdRadar, Ecu.eps) and car_model == CAR.HYUNDAI_GENESIS:
            continue
          if platform_code_ecu == Ecu.eps and car_model in no_eps_platforms:
            continue
          self.assertIn(platform_code_ecu, [e[0] for e in ecus])

  def test_fw_format(self):
    # Asserts:
    # - every supported ECU FW version returns one platform code
    # - every supported ECU FW version has a part number
    # - expected parsing of ECU FW dates

    for car_model, ecus in FW_VERSIONS.items():
      with self.subTest(car_model=car_model.value):
        for ecu, fws in ecus.items():
          if ecu[0] not in PLATFORM_CODE_ECUS:
            continue

          codes = set()
          for fw in fws:
            result = get_platform_codes([fw])
            self.assertEqual(1, len(result), f"Unable to parse FW: {fw}")
            codes |= result

          if ecu[0] not in DATE_FW_ECUS or car_model in NO_DATES_PLATFORMS:
            self.assertTrue(all(date is None for _, date in codes))
          else:
            self.assertTrue(all(date is not None for _, date in codes))

          if car_model == CAR.HYUNDAI_GENESIS:
            raise unittest.SkipTest("No part numbers for car model")

          # Hyundai places the ECU part number in their FW versions, assert all parsable
          # Some examples of valid formats: b"56310-L0010", b"56310L0010", b"56310/M6300"
          self.assertTrue(all(b"-" in code for code, _ in codes),
                          f"FW does not have part number: {fw}")

  def test_platform_codes_spot_check(self):
    # Asserts basic platform code parsing behavior for a few cases
    results = get_platform_codes([b"\xf1\x00DH LKAS 1.1 -150210"])
    self.assertEqual(results, {(b"DH", b"150210")})

    # Some cameras and all radars do not have dates
    results = get_platform_codes([b"\xf1\x00AEhe SCC H-CUP      1.01 1.01 96400-G2000         "])
    self.assertEqual(results, {(b"AEhe-G2000", None)})

    results = get_platform_codes([b"\xf1\x00CV1_ RDR -----      1.00 1.01 99110-CV000         "])
    self.assertEqual(results, {(b"CV1-CV000", None)})

    results = get_platform_codes([
      b"\xf1\x00DH LKAS 1.1 -150210",
      b"\xf1\x00AEhe SCC H-CUP      1.01 1.01 96400-G2000         ",
      b"\xf1\x00CV1_ RDR -----      1.00 1.01 99110-CV000         ",
    ])
    self.assertEqual(results, {(b"DH", b"150210"), (b"AEhe-G2000", None), (b"CV1-CV000", None)})

    results = get_platform_codes([
      b"\xf1\x00LX2 MFC  AT USA LHD 1.00 1.07 99211-S8100 220222",
      b"\xf1\x00LX2 MFC  AT USA LHD 1.00 1.08 99211-S8100 211103",
      b"\xf1\x00ON  MFC  AT USA LHD 1.00 1.01 99211-S9100 190405",
      b"\xf1\x00ON  MFC  AT USA LHD 1.00 1.03 99211-S9100 190720",
    ])
    self.assertEqual(results, {(b"LX2-S8100", b"220222"), (b"LX2-S8100", b"211103"),
                               (b"ON-S9100", b"190405"), (b"ON-S9100", b"190720")})

  def test_fuzzy_excluded_platforms(self):
    # Asserts a list of platforms that will not fuzzy fingerprint with platform codes due to them being shared.
    # This list can be shrunk as we combine platforms and detect features
    excluded_platforms = {
      CAR.GENESIS_G70,            # shared platform code, part number, and date
      CAR.GENESIS_G70_2020,
    }
    excluded_platforms |= CANFD_CAR - EV_CAR - CANFD_FUZZY_WHITELIST  # shared platform codes
    excluded_platforms |= NO_DATES_PLATFORMS  # date codes are required to match

    platforms_with_shared_codes = set()
    for platform, fw_by_addr in FW_VERSIONS.items():
      car_fw = []
      for ecu, fw_versions in fw_by_addr.items():
        ecu_name, addr, sub_addr = ecu
        for fw in fw_versions:
          car_fw.append({"ecu": ecu_name, "fwVersion": fw, "address": addr,
                         "subAddress": 0 if sub_addr is None else sub_addr})

      CP = car.CarParams.new_message(carFw=car_fw)
      matches = FW_QUERY_CONFIG.match_fw_to_car_fuzzy(build_fw_dict(CP.carFw), FW_VERSIONS)
      if len(matches) == 1:
        self.assertEqual(list(matches)[0], platform)
      else:
        platforms_with_shared_codes.add(platform)

    self.assertEqual(platforms_with_shared_codes, excluded_platforms)


if __name__ == "__main__":
  unittest.main()
