#!/usr/bin/env python3
import unittest
from typing import Dict, Iterable, Optional, Tuple

import capnp
from parameterized import parameterized
from hypothesis import settings, given, strategies as st

from cereal import car
from openpilot.selfdrive.car.ford.values import FW_QUERY_CONFIG, get_platform_codes
from openpilot.selfdrive.car.ford.fingerprints import FW_VERSIONS

Ecu = car.CarParams.Ecu


ECU_ADDRESSES = {
  Ecu.eps: 0x730,          # Power Steering Control Module (PSCM)
  Ecu.abs: 0x760,          # Anti-Lock Brake System (ABS)
  Ecu.fwdRadar: 0x764,     # Cruise Control Module (CCM)
  Ecu.fwdCamera: 0x706,    # Image Processing Module A (IPMA)
  Ecu.engine: 0x7E0,       # Powertrain Control Module (PCM)
  Ecu.shiftByWire: 0x732,  # Gear Shift Module (GSM)
}


ECU_FW_CORE = {
  Ecu.eps: [
    b"14D003",
  ],
  Ecu.abs: [
    b"2D053",
  ],
  Ecu.fwdRadar: [
    b"14D049",
  ],
  Ecu.fwdCamera: [
    b"14F397",  # Ford Q3
    b"14H102",  # Ford Q4
  ],
  Ecu.engine: [
    b"14C204",
  ],
}


class TestFordFW(unittest.TestCase):
  def test_fw_query_config(self):
    for (ecu, addr, subaddr) in FW_QUERY_CONFIG.extra_ecus:
      self.assertIn(ecu, ECU_ADDRESSES, "Unknown ECU")
      self.assertEqual(addr, ECU_ADDRESSES[ecu], "ECU address mismatch")
      self.assertIsNone(subaddr, "Unexpected ECU subaddress")

  @parameterized.expand(FW_VERSIONS.items())
  def test_fw_versions(self, car_model: str, fw_versions: Dict[Tuple[capnp.lib.capnp._EnumModule, int, Optional[int]], Iterable[bytes]]):
    for (ecu, addr, subaddr), fws in fw_versions.items():
      self.assertIn(ecu, ECU_ADDRESSES, "Unknown ECU")
      self.assertEqual(addr, ECU_ADDRESSES[ecu], "ECU address mismatch")
      self.assertIsNone(subaddr, "Unexpected ECU subaddress")

      for fw in fws:
        codes = get_platform_codes([fw])
        self.assertEqual(1, len(codes), f"Unable to parse FW: {fw!r}")

  @settings(max_examples=100)
  @given(data=st.data())
  def test_platform_codes_fuzzy_fw(self, data):
    """Ensure function doesn't raise an exception"""
    fw_strategy = st.lists(st.binary())
    fws = data.draw(fw_strategy)
    get_platform_codes(fws)


if __name__ == "__main__":
  unittest.main()
