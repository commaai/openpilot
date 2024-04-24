#!/usr/bin/env python3
from collections import defaultdict
import re
import unittest

from openpilot.selfdrive.car.honda.fingerprints import FW_VERSIONS
from openpilot.selfdrive.car.honda.fuzzy import HONDA_FW_PATTERN, PLATFORM_CODE_ECUS, get_platform_codes


class TestHondaFingerprint(unittest.TestCase):
  def test_fw_version_format(self):
    # Asserts all FW versions follow an expected format
    for fw_by_ecu in FW_VERSIONS.values():
      for fws in fw_by_ecu.values():
        for fw in fws:
          self.assertTrue(re.match(HONDA_FW_PATTERN, fw) is not None, fw)

  def test_no_shared_platform_codes(self):
    codes = defaultdict(lambda: set())
    for candidate, fw_by_ecu in FW_VERSIONS.items():
      for (ecu, _, _), fw_versions in fw_by_ecu.items():
        if ecu in PLATFORM_CODE_ECUS:
          for code in get_platform_codes(fw_versions):
            codes[code].add(candidate)

    for code, candidates in codes.items():
      with self.subTest(code=code):
        self.assertEqual(len(candidates), 1, f"Shared platform code {code} between {candidates}")


if __name__ == "__main__":
  unittest.main()
