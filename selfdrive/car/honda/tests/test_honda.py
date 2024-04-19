#!/usr/bin/env python3
import re
import unittest

from openpilot.selfdrive.car.honda.fingerprints import FW_VERSIONS
from openpilot.selfdrive.car.honda.fuzzy import HONDA_PARTNO_RE


class TestHondaFingerprint(unittest.TestCase):
  def test_fw_version_format(self):
    # Asserts all FW versions follow an expected format
    for fw_by_ecu in FW_VERSIONS.values():
      for fws in fw_by_ecu.values():
        for fw in fws:
          self.assertTrue(re.match(HONDA_PARTNO_RE, fw) is not None, fw)


if __name__ == "__main__":
  unittest.main()
