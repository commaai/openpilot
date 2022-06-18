#!/usr/bin/env python3
import os
import tempfile
import unittest
from parameterized import parameterized

import cereal.services as services
from cereal.services import service_list, RESERVED_PORT, STARTING_PORT


class TestServices(unittest.TestCase):

  @parameterized.expand(service_list.keys())
  def test_services(self, s):
    service = service_list[s]
    self.assertTrue(service.port != RESERVED_PORT)
    self.assertTrue(service.port >= STARTING_PORT)
    self.assertTrue(service.frequency <= 100)

  def test_no_duplicate_port(self):
    ports = {}
    for name, service in service_list.items():
      self.assertFalse(service.port in ports.keys(), f"duplicate port {service.port}")
      ports[service.port] = name

  def test_generated_header(self):
    with tempfile.NamedTemporaryFile(suffix=".h") as f:
      ret = os.system(f"python3 {services.__file__} > {f.name} && clang++ {f.name}")
      self.assertEqual(ret, 0, "generated services header is not valid C")

if __name__ == "__main__":
  unittest.main()
