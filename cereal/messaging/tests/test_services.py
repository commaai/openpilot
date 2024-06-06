#!/usr/bin/env python3
import os
import tempfile
from typing import Dict
import unittest
from parameterized import parameterized

import cereal.services as services
from cereal.services import SERVICE_LIST


class TestServices(unittest.TestCase):

  @parameterized.expand(SERVICE_LIST.keys())
  def test_services(self, s):
    service = SERVICE_LIST[s]
    self.assertTrue(service.frequency <= 104)

  def test_generated_header(self):
    with tempfile.NamedTemporaryFile(suffix=".h") as f:
      ret = os.system(f"python3 {services.__file__} > {f.name} && clang++ {f.name}")
      self.assertEqual(ret, 0, "generated services header is not valid C")

if __name__ == "__main__":
  unittest.main()
