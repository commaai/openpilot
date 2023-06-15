#!/usr/bin/env python3
import copy
import random
import time
import unittest
from collections import defaultdict
from parameterized import parameterized
import threading

from cereal import car
from common.params import Params
from selfdrive.car.car_helpers import interfaces
from selfdrive.car.fingerprints import FW_VERSIONS
from selfdrive.car.fw_versions import FW_QUERY_CONFIGS, FUZZY_EXCLUDE_ECUS, VERSIONS, build_fw_dict, match_fw_to_car, get_fw_versions

CarFw = car.CarParams.CarFw
Ecu = car.CarParams.Ecu

ECU_NAME = {v: k for k, v in Ecu.schema.enumerants.items()}
VERSIONS_COPY = copy.deepcopy(VERSIONS)


class FakeSocket:
  def receive(self, non_blocking=False):
    pass

  def send(self, msg):
    pass


class TestFwFingerprint(unittest.TestCase):
  def assertFingerprints(self, candidates, expected):
    candidates = list(candidates)
    self.assertEqual(len(candidates), 1, f"got more than one candidate: {candidates}")
    self.assertEqual(candidates[0], expected)

  @parameterized.expand([(b, c, e[c]) for b, e in VERSIONS.items() for c in e])
  def test_exact_match(self, brand, car_model, ecus):
    CP = car.CarParams.new_message()
    for _ in range(200):
      fw = []
      for ecu, fw_versions in ecus.items():
        ecu_name, addr, sub_addr = ecu
        fw.append({"ecu": ecu_name, "fwVersion": random.choice(fw_versions), 'brand': brand,
                   "address": addr, "subAddress": 0 if sub_addr is None else sub_addr})
      CP.carFw = fw
      _, matches = match_fw_to_car(CP.carFw, allow_fuzzy=False)
      self.assertFingerprints(matches, car_model)

  @parameterized.expand([(b, c, e[c]) for b, e in VERSIONS.items() for c in e])
  def test_custom_fuzzy_match(self, brand, car_model, ecus):
    # Assert brand-specific fuzzy fingerprinting function doesn't disagree with standard fuzzy function
    config = FW_QUERY_CONFIGS[brand]
    if config.match_fw_to_car_fuzzy is None:
      raise unittest.SkipTest("Brand does not implement custom fuzzy fingerprinting function")

class TestFwFingerprintLive(unittest.TestCase):
  @staticmethod
  def _benchmark(brand, num_pandas, n):
    params = Params()
    fake_socket = FakeSocket()

    times = []
    for _ in range(n):
      params.put_bool("ObdMultiplexingEnabled", True)
      thread = threading.Thread(target=get_fw_versions, args=(fake_socket, fake_socket, brand), kwargs=dict(num_pandas=num_pandas))
      thread.start()
      t = time.perf_counter()
      while thread.is_alive():
        time.sleep(0.02)
        if not params.get_bool("ObdMultiplexingChanged"):
          params.put_bool("ObdMultiplexingChanged", True)
      times.append(time.perf_counter() - t)

    return round(sum(times) / len(times), 2)

  def _assert_timing(self, avg_time, ref_time, tol):
    self.assertLess(avg_time, ref_time + tol)
    self.assertGreater(avg_time, ref_time - tol, "Performance seems to have improved, update test refs.")

  def test_fw_query_live(self):
    # Runs one of the functions used live to query FW versions from the car (get_fw_versions)
    # This test asserts a few things:
    # - Brand and whole query timings
    # - Global FW versions dict that is uses to query with and match candidates against remains static
    tol = 0.1
    total_ref_time = 4.6
    brand_ref_times = {
      1: {
        'body': 0.1,
        'chrysler': 0.3,
        'ford': 0.2,
        'honda': 0.5,
        'hyundai': 0.7,
        'mazda': 0.1,
        'nissan': 0.3,
        'subaru': 0.1,
        'tesla': 0.2,
        'toyota': 0.7,
        'volkswagen': 0.2,
      },
      2: {
        'hyundai': 1.1,
      }
    }

    total_time = 0
    for num_pandas in (1, 2):
      for brand, config in FW_QUERY_CONFIGS.items():
        with self.subTest(brand=brand, num_pandas=num_pandas):
          multi_panda_requests = [r for r in config.requests if r.bus > 3]
          if not len(multi_panda_requests) and num_pandas > 1:
            raise unittest.SkipTest("No multi-panda FW queries")

          avg_time = self._benchmark(brand, num_pandas, 5)
          total_time += avg_time
          self._assert_timing(avg_time, brand_ref_times[num_pandas][brand], tol)
          print(f'{brand=}, {num_pandas=}, {len(config.requests)=}, avg FW query time={avg_time} seconds')

    with self.subTest(brand='all_brands'):
      self._assert_timing(total_time, total_ref_time, tol)
      print(f'all brands, total FW query time={total_time} seconds')

    self.assertEqual(VERSIONS, VERSIONS_COPY, 'VERSIONS dictionary changed while fingerprinting')


if __name__ == "__main__":
  unittest.main()
