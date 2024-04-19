#!/usr/bin/env python3
from collections import defaultdict
import re
import unittest

from openpilot.selfdrive.car.fw_versions import ESSENTIAL_ECUS
from openpilot.selfdrive.car.honda.fingerprints import FW_VERSIONS
from openpilot.selfdrive.car.honda.fuzzy import get_platform_codes, HONDA_PARTNO_RE


class TestHondaFingerprint(unittest.TestCase):
  def test_fw_version_format(self):
    # Asserts all FW versions follow an expected format
    for fw_by_ecu in FW_VERSIONS.values():
      for fws in fw_by_ecu.values():
        for fw in fws:
          self.assertTrue(re.match(HONDA_PARTNO_RE, fw) is not None, fw)

  def test_no_overlap(self):
    # Asserts there is not two cars with identical part no + platform code
    codes = defaultdict(lambda: set())

    for fw_by_ecu in FW_VERSIONS.values():
      for ecu, fws in fw_by_ecu.items():
        if ecu in ESSENTIAL_ECUS:
          new_codes = set(get_platform_codes(fws).keys())
          self.assertTrue(codes[ecu].isdisjoint(new_codes), f"Overlapping codes: {ecu}: {codes[ecu].intersection(new_codes)}")
          codes[ecu].update(new_codes)


if __name__ == "__main__":
  unittest.main()
