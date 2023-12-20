#!/usr/bin/env python3
import re
import unittest

from openpilot.selfdrive.car.honda.fingerprints import FW_VERSIONS

HONDA_FW_VERSION_RE = br"\d{5}-[A-Z0-9]{3}(-|,)[A-Z0-9]{4}(\x00){2}$"


class TestHondaFingerprint(unittest.TestCase):
  def test_fw_version_format(self):
    # Asserts all FW versions follow an expected format
    for fw_by_ecu in FW_VERSIONS.values():
      for fws in fw_by_ecu.values():
        for fw in fws:
          self.assertTrue(re.match(HONDA_FW_VERSION_RE, fw) is not None, fw)


if __name__ == "__main__":
  unittest.main()
