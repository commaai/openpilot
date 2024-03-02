#!/usr/bin/env python3

import unittest
from openpilot.selfdrive.car import CarFlags
from openpilot.selfdrive.car.values import PLATFORMS


class TestPlatformConfigs(unittest.TestCase):
  def test_configs(self):

    for platform in PLATFORMS.values():
      with self.subTest(platform=str(platform)):
        self.assertTrue(platform.config._frozen)
        self.assertIn("pt", platform.config.dbc_dict)
        self.assertTrue(len(platform.config.platform_str) > 0)

        self.assertIsNotNone(platform.config.specs)



class SampleCarFlags(CarFlags):
  FLAG1 = 1
  FLAG2 = 2
  FLAG3 = 4


class TestCarFlags(unittest.TestCase):
  def test_is_set_and_any_set(self):

    for flag, is_set in [(0, False), (int(SampleCarFlags.FLAG1), True)]:
      self.assertEqual(SampleCarFlags.FLAG1.is_set(flag), is_set)

    # can't check an OR of flags with is_set
    with self.assertRaises(ValueError):
      (SampleCarFlags.FLAG1 | SampleCarFlags.FLAG2).is_set(0)

    # but you can with any_set
    self.assertTrue((SampleCarFlags.FLAG1 | SampleCarFlags.FLAG2).any_set(int(SampleCarFlags.FLAG1)))
    self.assertTrue((SampleCarFlags.FLAG1 | SampleCarFlags.FLAG2).any_set(int(SampleCarFlags.FLAG2)))


if __name__ == "__main__":
  unittest.main()
