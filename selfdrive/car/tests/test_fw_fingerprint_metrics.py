#!/usr/bin/env python3
import random
import unittest
from collections import defaultdict
from parameterized import parameterized

from cereal import car
from selfdrive.car.car_helpers import get_interface_attr, interfaces
from selfdrive.car.fingerprints import FW_VERSIONS
from selfdrive.car.fw_versions import FW_QUERY_CONFIGS, VERSIONS, match_fw_to_car

CarFw = car.CarParams.CarFw
Ecu = car.CarParams.Ecu

ECU_NAME = {v: k for k, v in Ecu.schema.enumerants.items()}


class TestFwFingerprint(unittest.TestCase):
  @parameterized.expand([(1,), (2,)])
  def test_fw_query_metrics(self, num_pandas):
    for brand, config in FW_QUERY_CONFIGS.items():
      with self.subTest(brand=brand, num_pandas=num_pandas):
        requests = [r for r in config.requests if r.bus <= num_pandas * 4 - 1]
        multi_panda_requests = [r for r in config.requests if r.bus > 3]

        if not len(multi_panda_requests) and num_pandas > 1:
          raise unittest.SkipTest("No multi-panda FW queries")

        total_time = 0
        obd_multiplexing = config.requests[0].obd_multiplexing  # only count OBD multiplexing transitions
        for r in requests:
          total_time += 0.1

          # if r.obd_multiplexing != obd_multiplexing and r.bus % 4 == 1:
          #   obd_multiplexing = r.obd_multiplexing
          #   total_time += 0.1

        total_time = round(total_time, 2)
        self.assertLessEqual(total_time, 1.1)
        print(f'{brand=}, {num_pandas=}, {len(requests)=}, total FW query time={total_time} seconds')


if __name__ == "__main__":
  unittest.main()
