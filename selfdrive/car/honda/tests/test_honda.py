#!/usr/bin/env python3
import re
import unittest

from cereal import car
from selfdrive.car.car_helpers import get_interface_attr
from selfdrive.car.honda.values import FW_VERSIONS

Ecu = car.CarParams.Ecu

ECU_NAME = {v: k for k, v in Ecu.schema.enumerants.items()}
VERSIONS = get_interface_attr("FW_VERSIONS", ignore_none=True)


HONDA_FW_VERSION_RE = br"\d{5}-[A-Z0-9]{3}(-|,)[A-Z0-9]{4}(\x00){2}$"


class TestHondaFingerprint(unittest.TestCase):
  def test_fw_version_format(self):
    # Asserts all FW versions follow a consistent format
    for car, fw_by_ecu in FW_VERSIONS.items():
      for ecu_type, fws in fw_by_ecu.items():
        for fw in fws:
          self.assertTrue(re.match(HONDA_FW_VERSION_RE, fw) is not None, fw)


if __name__ == "__main__":
  unittest.main()
