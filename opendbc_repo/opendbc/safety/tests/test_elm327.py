#!/usr/bin/env python3
import unittest

import opendbc.safety.tests.common as common
from opendbc.car.structs import CarParams
from opendbc.safety import DLC_TO_LEN
from opendbc.safety.tests.libsafety import libsafety_py
from opendbc.safety.tests.test_defaults import TestDefaultRxHookBase

GM_CAMERA_DIAG_ADDR = 0x24B


class TestElm327(TestDefaultRxHookBase):
  TX_MSGS = [[addr, bus] for addr in [GM_CAMERA_DIAG_ADDR, *range(0x600, 0x800),
                                      *range(0x18DA00F1, 0x18DB00F1, 0x100),  # 29-bit UDS physical addressing
                                      *[0x18DB33F1],  # 29-bit UDS functional address
                                      ] for bus in range(4)]
  FWD_BUS_LOOKUP = {}

  def setUp(self):
    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(CarParams.SafetyModel.elm327, 0)
    self.safety.init_tests()

  def test_tx_hook(self):
    # ensure we can transmit arbitrary data on allowed addresses
    for bus in range(4):
      for addr in self.SCANNED_ADDRS:
        should_tx = [addr, bus] in self.TX_MSGS
        self.assertEqual(should_tx, self._tx(common.make_msg(bus, addr, 8)))

    # ELM only allows 8 byte UDS/KWP messages under ISO 15765-4
    for msg_len in DLC_TO_LEN:
      should_tx = msg_len == 8
      self.assertEqual(should_tx, self._tx(common.make_msg(0, 0x700, msg_len)))

    # TODO: perform this check for all addresses
    # 4 to 15 are reserved ISO-TP frame types (https://en.wikipedia.org/wiki/ISO_15765-2)
    for byte in range(0xff):
      should_tx = (byte >> 4) <= 3
      self.assertEqual(should_tx, self._tx(common.make_msg(0, GM_CAMERA_DIAG_ADDR, dat=bytes([byte] * 8))))

    # test GM camera diagnostic address with malformed length
    self.assertEqual(False, self._tx(common.make_msg(0, GM_CAMERA_DIAG_ADDR, dat=bytes([0x00] * 7))))

  def test_tx_hook_on_wrong_safety_mode(self):
    # No point, since we allow many diagnostic addresses
    pass


if __name__ == "__main__":
  unittest.main()
