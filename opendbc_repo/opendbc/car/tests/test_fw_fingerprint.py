import pytest
import random
import time
from collections import defaultdict
from parameterized import parameterized

from opendbc.car.can_definitions import CanData
from opendbc.car.car_helpers import interfaces
from opendbc.car.structs import CarParams
from opendbc.car.fingerprints import FW_VERSIONS
from opendbc.car.fw_versions import ESSENTIAL_ECUS, FW_QUERY_CONFIGS, FUZZY_EXCLUDE_ECUS, VERSIONS, build_fw_dict, \
                                                match_fw_to_car, get_brand_ecu_matches, get_fw_versions, get_present_ecus
from opendbc.car.vin import get_vin

CarFw = CarParams.CarFw
Ecu = CarParams.Ecu


class TestFwFingerprint:
  def assertFingerprints(self, candidates, expected):
    candidates = list(candidates)
    assert len(candidates) == 1, f"got more than one candidate: {candidates}"
    assert candidates[0] == expected

  @parameterized.expand([(b, c, e[c], n) for b, e in VERSIONS.items() for c in e for n in (True, False)])
  def test_exact_match(self, brand, car_model, ecus, test_non_essential):
    config = FW_QUERY_CONFIGS[brand]
    CP = CarParams()
    for _ in range(20):
      fw = []
      for ecu, fw_versions in ecus.items():
        # Assume non-essential ECUs apply to all cars, so we catch cases where Car A with
        # missing ECUs won't match to Car B where only Car B has labeled non-essential ECUs
        if ecu[0] in config.non_essential_ecus and test_non_essential:
          continue

        ecu_name, addr, sub_addr = ecu
        fw.append(CarFw(ecu=ecu_name, fwVersion=random.choice(fw_versions), brand=brand,
                        address=addr, subAddress=0 if sub_addr is None else sub_addr))
      CP.carFw = fw
      _, matches = match_fw_to_car(CP.carFw, CP.carVin, allow_fuzzy=False)
      if not test_non_essential:
        self.assertFingerprints(matches, car_model)
      else:
        # if we're removing ECUs we expect some match loss, but it shouldn't mismatch
        if len(matches) != 0:
          self.assertFingerprints(matches, car_model)

  @parameterized.expand([(b, c, e[c]) for b, e in VERSIONS.items() for c in e])
  def test_custom_fuzzy_match(self, brand, car_model, ecus):
    # Assert brand-specific fuzzy fingerprinting function doesn't disagree with standard fuzzy function
    config = FW_QUERY_CONFIGS[brand]
    if config.match_fw_to_car_fuzzy is None:
      pytest.skip("Brand does not implement custom fuzzy fingerprinting function")

    CP = CarParams()
    for _ in range(5):
      fw = []
      for ecu, fw_versions in ecus.items():
        ecu_name, addr, sub_addr = ecu
        fw.append(CarFw(ecu=ecu_name, fwVersion=random.choice(fw_versions), brand=brand,
                        address=addr, subAddress=0 if sub_addr is None else sub_addr))
      CP.carFw = fw
      _, matches = match_fw_to_car(CP.carFw, CP.carVin, allow_exact=False, log=False)
      brand_matches = config.match_fw_to_car_fuzzy(build_fw_dict(CP.carFw), CP.carVin, VERSIONS[brand])

      # If both have matches, they must agree
      if len(matches) == 1 and len(brand_matches) == 1:
        assert matches == brand_matches

  @parameterized.expand([(b, c, e[c]) for b, e in VERSIONS.items() for c in e])
  def test_fuzzy_match_ecu_count(self, brand, car_model, ecus):
    # Asserts that fuzzy matching does not count matching FW, but ECU address keys
    valid_ecus = [e for e in ecus if e[0] not in FUZZY_EXCLUDE_ECUS]
    if not len(valid_ecus):
      pytest.skip("Car model has no compatible ECUs for fuzzy matching")

    fw = []
    for ecu in valid_ecus:
      ecu_name, addr, sub_addr = ecu
      for _ in range(5):
        # Add multiple FW versions to simulate ECU returning to multiple queries in a brand
        fw.append(CarFw(ecu=ecu_name, fwVersion=random.choice(ecus[ecu]), brand=brand,
                        address=addr, subAddress=0 if sub_addr is None else sub_addr))
      CP = CarParams(carFw=fw)
      _, matches = match_fw_to_car(CP.carFw, CP.carVin, allow_exact=False, log=False)

      # Assert no match if there are not enough unique ECUs
      unique_ecus = {(f.address, f.subAddress) for f in fw}
      if len(unique_ecus) < 2:
        assert len(matches) == 0, car_model
      # There won't always be a match due to shared FW, but if there is it should be correct
      elif len(matches):
        self.assertFingerprints(matches, car_model)

  def test_fw_version_lists(self, subtests):
    for car_model, ecus in FW_VERSIONS.items():
      with subtests.test(car_model=car_model.value):
        for ecu, ecu_fw in ecus.items():
          with subtests.test(ecu):
            duplicates = {fw for fw in ecu_fw if ecu_fw.count(fw) > 1}
            assert not len(duplicates), f'{car_model}: Duplicate FW versions: Ecu.{ecu[0]}, {duplicates}'
            assert len(ecu_fw) > 0, f'{car_model}: No FW versions: Ecu.{ecu[0]}'

  def test_all_addrs_map_to_one_ecu(self):
    for brand, cars in VERSIONS.items():
      addr_to_ecu = defaultdict(set)
      for ecus in cars.values():
        for ecu_type, addr, sub_addr in ecus.keys():
          addr_to_ecu[(addr, sub_addr)].add(ecu_type)
          ecus_for_addr = addr_to_ecu[(addr, sub_addr)]
          ecu_strings = ", ".join([f'Ecu.{ecu}' for ecu in ecus_for_addr])
          assert len(ecus_for_addr) <= 1, f"{brand} has multiple ECUs that map to one address: {ecu_strings} -> ({hex(addr)}, {sub_addr})"

  def test_data_collection_ecus(self, subtests):
    # Asserts no extra ECUs are in the fingerprinting database
    for brand, config in FW_QUERY_CONFIGS.items():
      for car_model, ecus in VERSIONS[brand].items():
        bad_ecus = set(ecus).intersection(config.extra_ecus)
        with subtests.test(car_model=car_model.value):
          assert not len(bad_ecus), f'{car_model}: Fingerprints contain ECUs added for data collection: {bad_ecus}'

  def test_blacklisted_ecus(self, subtests):
    blacklisted_addrs = (0x7c4, 0x7d0)  # includes A/C ecu and an unknown ecu
    for car_model, ecus in FW_VERSIONS.items():
      with subtests.test(car_model=car_model.value):
        CP = interfaces[car_model][0].get_non_essential_params(car_model)
        if CP.carName == 'subaru':
          for ecu in ecus.keys():
            assert ecu[1] not in blacklisted_addrs, f'{car_model}: Blacklisted ecu: (Ecu.{ecu[0]}, {hex(ecu[1])})'

        elif CP.carName == "chrysler":
          # Some HD trucks have a combined TCM and ECM
          if CP.carFingerprint.startswith("RAM_HD"):
            for ecu in ecus.keys():
              assert ecu[0] != Ecu.transmission, f"{car_model}: Blacklisted ecu: (Ecu.{ecu[0]}, {hex(ecu[1])})"

  def test_non_essential_ecus(self, subtests):
    for brand, config in FW_QUERY_CONFIGS.items():
      with subtests.test(brand):
        # These ECUs are already not in ESSENTIAL_ECUS which the fingerprint functions give a pass if missing
        unnecessary_non_essential_ecus = set(config.non_essential_ecus) - set(ESSENTIAL_ECUS)
        assert unnecessary_non_essential_ecus == set(), "Declaring non-essential ECUs non-essential is not required: " + \
                                                                f"{', '.join([f'Ecu.{ecu}' for ecu in unnecessary_non_essential_ecus])}"

  def test_missing_versions_and_configs(self, subtests):
    brand_versions = set(VERSIONS.keys())
    brand_configs = set(FW_QUERY_CONFIGS.keys())
    if len(brand_configs - brand_versions):
      with subtests.test():
        pytest.fail(f"Brands do not implement FW_VERSIONS: {brand_configs - brand_versions}")

    if len(brand_versions - brand_configs):
      with subtests.test():
        pytest.fail(f"Brands do not implement FW_QUERY_CONFIG: {brand_versions - brand_configs}")

    # Ensure each brand has at least 1 ECU to query, and extra ECU retrieval
    for brand, config in FW_QUERY_CONFIGS.items():
      assert len(config.get_all_ecus({}, include_extra_ecus=False)) == 0
      assert config.get_all_ecus({}) == set(config.extra_ecus)
      assert len(config.get_all_ecus(VERSIONS[brand])) > 0

  def test_fw_request_ecu_whitelist(self, subtests):
    for brand, config in FW_QUERY_CONFIGS.items():
      with subtests.test(brand=brand):
        whitelisted_ecus = {ecu for r in config.requests for ecu in r.whitelist_ecus}
        brand_ecus = {fw[0] for car_fw in VERSIONS[brand].values() for fw in car_fw}
        brand_ecus |= {ecu[0] for ecu in config.extra_ecus}

        # each ecu in brand's fw versions + extra ecus needs to be whitelisted at least once
        ecus_not_whitelisted = brand_ecus - whitelisted_ecus

        ecu_strings = ", ".join([f'Ecu.{ecu}' for ecu in ecus_not_whitelisted])
        assert not (len(whitelisted_ecus) and len(ecus_not_whitelisted)), \
                         f'{brand.title()}: ECUs not in any FW query whitelists: {ecu_strings}'

  def test_fw_requests(self, subtests):
    # Asserts equal length request and response lists
    for brand, config in FW_QUERY_CONFIGS.items():
      with subtests.test(brand=brand):
        for request_obj in config.requests:
          assert len(request_obj.request) == len(request_obj.response)

          # No request on the OBD port (bus 1, multiplexed) should be run on an aux panda
          assert not (request_obj.auxiliary and request_obj.bus == 1 and request_obj.obd_multiplexing), \
                           f"{brand.title()}: OBD multiplexed request is marked auxiliary: {request_obj}"

  def test_brand_ecu_matches(self):
    empty_response = {brand: set() for brand in FW_QUERY_CONFIGS}
    assert get_brand_ecu_matches(set()) == empty_response

    # we ignore bus
    expected_response = empty_response | {'toyota': {(0x750, 0xf)}}
    assert get_brand_ecu_matches({(0x758, 0xf, 99)}) == expected_response


class TestFwFingerprintTiming:
  N: int = 5
  TOL: float = 0.05

  # for patched functions
  current_obd_multiplexing: bool
  total_time: float

  @staticmethod
  def fake_can_send(msgs):
    pass

  @staticmethod
  def fake_can_recv(wait_for_one: bool = False) -> list[list[CanData]]:
    return ([[CanData(random.randint(0x600, 0x800), b'\x00' * 8, 0)]]
            if random.uniform(0, 1) > 0.5 else [])

  def fake_set_obd_multiplexing(self, obd_multiplexing):
    """The 10Hz blocking params loop adds on average 50ms to the query time for each OBD multiplexing change"""
    if obd_multiplexing != self.current_obd_multiplexing:
      self.current_obd_multiplexing = obd_multiplexing
      self.total_time += 0.1 / 2

  def fake_get_data(self, timeout):
    self.total_time += timeout
    return {}

  def _benchmark_brand(self, brand, num_pandas, mocker):
    self.total_time = 0
    mocker.patch("opendbc.car.isotp_parallel_query.IsoTpParallelQuery.get_data", self.fake_get_data)
    for _ in range(self.N):
      # Treat each brand as the most likely (aka, the first) brand with OBD multiplexing initially on
      self.current_obd_multiplexing = True

      t = time.perf_counter()
      get_fw_versions(self.fake_can_recv, self.fake_can_send, self.fake_set_obd_multiplexing, brand, num_pandas=num_pandas)
      self.total_time += time.perf_counter() - t

    return self.total_time / self.N

  def _assert_timing(self, avg_time, ref_time):
    assert avg_time < ref_time + self.TOL
    assert avg_time > ref_time - self.TOL, "Performance seems to have improved, update test refs."

  def test_startup_timing(self, subtests, mocker):
    # Tests worse-case VIN query time and typical present ECU query time
    vin_ref_times = {'worst': 1.4, 'best': 0.7}  # best assumes we go through all queries to get a match
    present_ecu_ref_time = 0.45

    def fake_get_ecu_addrs(*_, timeout):
      self.total_time += timeout
      return set()

    self.total_time = 0.0
    mocker.patch("opendbc.car.fw_versions.get_ecu_addrs", fake_get_ecu_addrs)
    for _ in range(self.N):
      self.current_obd_multiplexing = True
      get_present_ecus(self.fake_can_recv, self.fake_can_send, self.fake_set_obd_multiplexing, num_pandas=2)
    self._assert_timing(self.total_time / self.N, present_ecu_ref_time)
    print(f'get_present_ecus, query time={self.total_time / self.N} seconds')

    for name, args in (('worst', {}), ('best', {'retry': 1})):
      with subtests.test(name=name):
        self.total_time = 0.0
        mocker.patch("opendbc.car.isotp_parallel_query.IsoTpParallelQuery.get_data", self.fake_get_data)
        for _ in range(self.N):
          get_vin(self.fake_can_recv, self.fake_can_send, (0, 1), **args)
        self._assert_timing(self.total_time / self.N, vin_ref_times[name])
        print(f'get_vin {name} case, query time={self.total_time / self.N} seconds')

  def test_fw_query_timing(self, subtests, mocker):
    total_ref_time = {1: 7.0, 2: 7.6}
    brand_ref_times = {
      1: {
        'gm': 1.0,
        'body': 0.1,
        'chrysler': 0.3,
        'ford': 1.5,
        'honda': 0.45,
        'hyundai': 0.65,
        'mazda': 0.1,
        'nissan': 0.8,
        'subaru': 0.65,
        'tesla': 0.1,
        'toyota': 0.7,
        'volkswagen': 0.65,
      },
      2: {
        'ford': 1.6,
        'hyundai': 1.15,
      }
    }

    total_times = {1: 0.0, 2: 0.0}
    for num_pandas in (1, 2):
      for brand, config in FW_QUERY_CONFIGS.items():
        with subtests.test(brand=brand, num_pandas=num_pandas):
          avg_time = self._benchmark_brand(brand, num_pandas, mocker)
          total_times[num_pandas] += avg_time
          avg_time = round(avg_time, 2)

          ref_time = brand_ref_times[num_pandas].get(brand)
          if ref_time is None:
            # ref time should be same as 1 panda if no aux queries
            ref_time = brand_ref_times[num_pandas - 1][brand]

          self._assert_timing(avg_time, ref_time)
          print(f'{brand=}, {num_pandas=}, {len(config.requests)=}, avg FW query time={avg_time} seconds')

    for num_pandas in (1, 2):
      with subtests.test(brand='all_brands', num_pandas=num_pandas):
        total_time = round(total_times[num_pandas], 2)
        self._assert_timing(total_time, total_ref_time[num_pandas])
        print(f'all brands, total FW query time={total_time} seconds')

  def test_get_fw_versions(self, subtests, mocker):
    # some coverage on IsoTpParallelQuery and panda UDS library
    # TODO: replace this with full fingerprint simulation testing
    # https://github.com/commaai/panda/pull/1329

    def fake_carlog_exception(*args, **kwargs):
      raise

    t = 0

    def fake_monotonic():
      nonlocal t
      t += 0.0001
      return t

    mocker.patch("opendbc.car.carlog.exception", fake_carlog_exception)
    mocker.patch("time.monotonic", fake_monotonic)
    for brand in FW_QUERY_CONFIGS.keys():
      with subtests.test(brand=brand):
        get_fw_versions(self.fake_can_recv, self.fake_can_send, lambda obd: None, brand)
