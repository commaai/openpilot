#!/usr/bin/env python3

import unittest

from openpilot.selfdrive.car.values import PLATFORMS


class TestPlatformConfigs(unittest.TestCase):
  def test_configs(self):

    for name, platform in PLATFORMS.items():
      with self.subTest(platform=str(platform)):
        self.assertTrue(platform.config._frozen)

        if platform != "MOCK":
          self.assertIn("pt", platform.config.dbc_dict)
        self.assertTrue(len(platform.config.platform_str) > 0)

        self.assertEqual(name, platform.config.platform_str)

        self.assertIsNotNone(platform.config.specs)


if __name__ == "__main__":
  unittest.main()
