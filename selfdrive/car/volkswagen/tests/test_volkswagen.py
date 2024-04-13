#!/usr/bin/env python3
import re
import unittest

from openpilot.selfdrive.car.volkswagen.fingerprints import FW_VERSIONS

# TODO: determine the unknown groups
SPARE_PART_FW_PATTERN = re.compile(b'\xf1\x87(?P<gateway>[0-9][0-9A-Z]{2})(?P<unknown>[0-9][0-9A-Z][0-9])(?P<unknown2>[0-9A-Z]{2}[0-9])([A-Z0-9]| )')


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
