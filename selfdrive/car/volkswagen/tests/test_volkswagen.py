#!/usr/bin/env python3
import unittest

from openpilot.selfdrive.car.volkswagen.values import SPARE_PART_FW_PATTERN
from openpilot.selfdrive.car.volkswagen.fingerprints import FW_VERSIONS


class TestVolkswagenPlatformConfigs(unittest.TestCase):
  def test_spare_part_fw_pattern(self):
    # Relied on for determining if a FW is likely VW
    for platform, ecus in FW_VERSIONS.items():
      with self.subTest(platform=platform):
        for fws in ecus.values():
          for fw in fws:
            self.assertNotEqual(SPARE_PART_FW_PATTERN.match(fw), None, f"Bad FW: {fw}")


if __name__ == "__main__":
  unittest.main()
