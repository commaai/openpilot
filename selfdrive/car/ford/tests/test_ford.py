#!/usr/bin/env python3
from collections import defaultdict
import unittest

from selfdrive.car.ford.values import FW_QUERY_CONFIG, FW_VERSIONS


class TestFordFingerprint(unittest.TestCase):
  def test_fuzzy_ecus_available(self):
    # Asserts ECU keys essential for fuzzy fingerprinting are available on all platforms
    for car_model, ecus in FW_VERSIONS.items():
      with self.subTest(car_model=car_model):
        for fuzzy_ecu in FW_QUERY_CONFIG.fuzzy_ecus:
          self.assertIn(fuzzy_ecu, [e[0] for e in ecus])

  def test_fuzzy_platform_codes(self):
    codes = FW_QUERY_CONFIG.fuzzy_get_platform_codes([
      b'L1MC-14D003-AJ\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'L1MC-14D003-AK\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'L1MC-14D003-AL\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'M1MC-14D003-AB\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ])
    self.assertEqual(codes, {b'1MC-AE', b'1MC-AH', b'1MC-AC', b'1MC-AK',
                             b'1MC-AG', b'1MC-AL', b'1MC-AD', b'1MC-AI',
                             b'1MC-AJ', b'1MC-AF', b'1MC-AB'})


if __name__ == "__main__":
  unittest.main()
