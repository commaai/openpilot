#!/usr/bin/env python3
import unittest
from parameterized import parameterized
from collections.abc import Iterable

import capnp

from cereal import car
from openpilot.selfdrive.car.ford.values import FW_QUERY_CONFIG
from openpilot.selfdrive.car.ford.fingerprints import FW_VERSIONS

Ecu = car.CarParams.Ecu


ECU_ADDRESSES = {
  Ecu.eps: 0x730,          # Power Steering Control Module (PSCM)
  Ecu.abs: 0x760,          # Anti-Lock Brake System (ABS)
  Ecu.fwdRadar: 0x764,     # Cruise Control Module (CCM)
  Ecu.fwdCamera: 0x706,    # Image Processing Module A (IPMA)
  Ecu.engine: 0x7E0,       # Powertrain Control Module (PCM)
  Ecu.shiftByWire: 0x732,  # Gear Shift Module (GSM)
  Ecu.debug: 0x7D0,        # Accessory Protocol Interface Module (APIM)
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
  def test_fw_versions(self, car_model: str, fw_versions: dict[tuple[capnp.lib.capnp._EnumModule, int, int | None], Iterable[bytes]]):
    for (ecu, addr, subaddr), fws in fw_versions.items():
      self.assertIn(ecu, ECU_FW_CORE, "Unexpected ECU")
      self.assertEqual(addr, ECU_ADDRESSES[ecu], "ECU address mismatch")
      self.assertIsNone(subaddr, "Unexpected ECU subaddress")

      # Software part number takes the form: PREFIX-CORE-SUFFIX
      # Prefix changes based on the family of part. It includes the model year
      #   and likely the platform.
      # Core identifies the type of the item (e.g. 14D003 = PSCM, 14C204 = PCM).
      # Suffix specifies the version of the part. -AA would be followed by -AB.
      #   Small increments in the suffix are usually compatible.
      # Details: https://forscan.org/forum/viewtopic.php?p=70008#p70008
      for fw in fws:
        self.assertEqual(len(fw), 24, "Expected ECU response to be 24 bytes")

        # TODO: parse with regex, don't need detailed error message
        fw_parts = fw.rstrip(b'\x00').split(b'-')
        self.assertEqual(len(fw_parts), 3, "Expected FW to be in format: prefix-core-suffix")

        prefix, core, suffix = fw_parts
        self.assertEqual(len(prefix), 4, "Expected FW prefix to be 4 characters")
        self.assertIn(len(core), (5, 6), "Expected FW core to be 5-6 characters")
        self.assertIn(core, ECU_FW_CORE[ecu], f"Unexpected FW core for {ecu}")
        self.assertIn(len(suffix), (2, 3), "Expected FW suffix to be 2-3 characters")


if __name__ == "__main__":
  unittest.main()
