from hypothesis import given, settings, strategies as st

from cereal import car
from openpilot.selfdrive.car.fw_versions import build_fw_dict
from openpilot.selfdrive.car.toyota.fingerprints import FW_VERSIONS
from openpilot.selfdrive.car.toyota.values import CAR, DBC, TSS2_CAR, ANGLE_CONTROL_CAR, RADAR_ACC_CAR, \
                                                  FW_QUERY_CONFIG, PLATFORM_CODE_ECUS, FUZZY_EXCLUDED_PLATFORMS, \
                                                  get_platform_codes

Ecu = car.CarParams.Ecu
ECU_NAME = {v: k for k, v in Ecu.schema.enumerants.items()}


def check_fw_version(fw_version: bytes) -> bool:
  # TODO: just use the FW patterns, need to support all chunks
  return b'?' not in fw_version and b'!' not in fw_version


class TestToyotaInterfaces:
  def test_car_sets(self):
    assert len(ANGLE_CONTROL_CAR - TSS2_CAR) == 0
    assert len(RADAR_ACC_CAR - TSS2_CAR) == 0

  def test_lta_platforms(self):
    # At this time, only RAV4 2023 is expected to use LTA/angle control
    assert ANGLE_CONTROL_CAR == {CAR.TOYOTA_RAV4_TSS2_2023}

  def test_tss2_dbc(self):
    # We make some assumptions about TSS2 platforms,
    # like looking up certain signals only in this DBC
    for car_model, dbc in DBC.items():
      if car_model in TSS2_CAR:
        assert dbc["pt"] == "toyota_nodsu_pt_generated"

  def test_essential_ecus(self, subtests):
    # Asserts standard ECUs exist for each platform
    common_ecus = {Ecu.fwdRadar, Ecu.fwdCamera}
    for car_model, ecus in FW_VERSIONS.items():
      with subtests.test(car_model=car_model.value):
        present_ecus = {ecu[0] for ecu in ecus}
        missing_ecus = common_ecus - present_ecus
        assert len(missing_ecus) == 0

        # Some exceptions for other common ECUs
        if car_model not in (CAR.TOYOTA_ALPHARD_TSS2,):
          assert Ecu.abs in present_ecus

        if car_model not in (CAR.TOYOTA_MIRAI,):
          assert Ecu.engine in present_ecus

        if car_model not in (CAR.TOYOTA_PRIUS_V, CAR.LEXUS_CTH):
          assert Ecu.eps in present_ecus


class TestToyotaFingerprint:
  def test_non_essential_ecus(self, subtests):
    # Ensures only the cars that have multiple engine ECUs are in the engine non-essential ECU list
    for car_model, ecus in FW_VERSIONS.items():
      with subtests.test(car_model=car_model.value):
        engine_ecus = {ecu for ecu in ecus if ecu[0] == Ecu.engine}
        assert (len(engine_ecus) > 1) == (car_model in FW_QUERY_CONFIG.non_essential_ecus[Ecu.engine]), \
          f"Car model unexpectedly {'not ' if len(engine_ecus) > 1 else ''}in non-essential list"

  def test_valid_fw_versions(self, subtests):
    # Asserts all FW versions are valid
    for car_model, ecus in FW_VERSIONS.items():
      with subtests.test(car_model=car_model.value):
        for fws in ecus.values():
          for fw in fws:
            assert check_fw_version(fw), fw

  # Tests for part numbers, platform codes, and sub-versions which Toyota will use to fuzzy
  # fingerprint in the absence of full FW matches:
  @settings(max_examples=100)
  @given(data=st.data())
  def test_platform_codes_fuzzy_fw(self, data):
    fw_strategy = st.lists(st.binary())
    fws = data.draw(fw_strategy)
    get_platform_codes(fws)

  def test_platform_code_ecus_available(self, subtests):
    # Asserts ECU keys essential for fuzzy fingerprinting are available on all platforms
    for car_model, ecus in FW_VERSIONS.items():
      with subtests.test(car_model=car_model.value):
        for platform_code_ecu in PLATFORM_CODE_ECUS:
          if platform_code_ecu == Ecu.eps and car_model in (CAR.TOYOTA_PRIUS_V, CAR.LEXUS_CTH,):
            continue
          if platform_code_ecu == Ecu.abs and car_model in (CAR.TOYOTA_ALPHARD_TSS2,):
            continue
          assert platform_code_ecu in [e[0] for e in ecus]

  def test_fw_format(self, subtests):
    # Asserts:
    # - every supported ECU FW version returns one platform code
    # - every supported ECU FW version has a part number
    # - expected parsing of ECU sub-versions

    for car_model, ecus in FW_VERSIONS.items():
      with subtests.test(car_model=car_model.value):
        for ecu, fws in ecus.items():
          if ecu[0] not in PLATFORM_CODE_ECUS:
            continue

          codes = dict()
          for fw in fws:
            result = get_platform_codes([fw])
            # Check only one platform code and sub-version
            assert 1 == len(result), f"Unable to parse FW: {fw}"
            assert 1 == len(list(result.values())[0]), f"Unable to parse FW: {fw}"
            codes |= result

          # Toyota places the ECU part number in their FW versions, assert all parsable
          # Note that there is only one unique part number per ECU across the fleet, so this
          # is not important for identification, just a sanity check.
          assert all(code.count(b"-") > 1 for code in codes), f"FW does not have part number: {fw} {codes}"

  def test_platform_codes_spot_check(self):
    # Asserts basic platform code parsing behavior for a few cases
    results = get_platform_codes([
      b"F152607140\x00\x00\x00\x00\x00\x00",
      b"F152607171\x00\x00\x00\x00\x00\x00",
      b"F152607110\x00\x00\x00\x00\x00\x00",
      b"F152607180\x00\x00\x00\x00\x00\x00",
    ])
    assert results == {b"F1526-07-1": {b"10", b"40", b"71", b"80"}}

    results = get_platform_codes([
      b"\x028646F4104100\x00\x00\x00\x008646G5301200\x00\x00\x00\x00",
      b"\x028646F4104100\x00\x00\x00\x008646G3304000\x00\x00\x00\x00",
    ])
    assert results == {b"8646F-41-04": {b"100"}}

    # Short version has no part number
    results = get_platform_codes([
      b"\x0235870000\x00\x00\x00\x00\x00\x00\x00\x00A0202000\x00\x00\x00\x00\x00\x00\x00\x00",
      b"\x0235883000\x00\x00\x00\x00\x00\x00\x00\x00A0202000\x00\x00\x00\x00\x00\x00\x00\x00",
    ])
    assert results == {b"58-70": {b"000"}, b"58-83": {b"000"}}

    results = get_platform_codes([
      b"F152607110\x00\x00\x00\x00\x00\x00",
      b"F152607140\x00\x00\x00\x00\x00\x00",
      b"\x028646F4104100\x00\x00\x00\x008646G5301200\x00\x00\x00\x00",
      b"\x0235879000\x00\x00\x00\x00\x00\x00\x00\x00A4701000\x00\x00\x00\x00\x00\x00\x00\x00",
    ])
    assert results == {b"F1526-07-1": {b"10", b"40"}, b"8646F-41-04": {b"100"}, b"58-79": {b"000"}}

  def test_fuzzy_excluded_platforms(self):
    # Asserts a list of platforms that will not fuzzy fingerprint with platform codes due to them being shared.
    platforms_with_shared_codes = set()
    for platform, fw_by_addr in FW_VERSIONS.items():
      car_fw = []
      for ecu, fw_versions in fw_by_addr.items():
        ecu_name, addr, sub_addr = ecu
        for fw in fw_versions:
          car_fw.append({"ecu": ecu_name, "fwVersion": fw, "address": addr,
                         "subAddress": 0 if sub_addr is None else sub_addr})

      CP = car.CarParams.new_message(carFw=car_fw)
      matches = FW_QUERY_CONFIG.match_fw_to_car_fuzzy(build_fw_dict(CP.carFw), CP.carVin, FW_VERSIONS)
      if len(matches) == 1:
        assert list(matches)[0] == platform
      else:
        # If a platform has multiple matches, add it and its matches
        platforms_with_shared_codes |= {str(platform), *matches}

    assert platforms_with_shared_codes == FUZZY_EXCLUDED_PLATFORMS, (len(platforms_with_shared_codes), len(FW_VERSIONS))
