#!/usr/bin/env python3
import unittest

import panda.tests.safety.common as common

from panda import DLC_TO_LEN, Panda
from panda.tests.libpanda import libpanda_py
from panda.tests.safety.test_defaults import TestDefaultRxHookBase

GM_CAMERA_DIAG_ADDR = 0x24B


class TestElm327(TestDefaultRxHookBase):
  TX_MSGS = [[addr, bus] for addr in [GM_CAMERA_DIAG_ADDR, *range(0x600, 0x800),
                                      *range(0x18DA00F1, 0x18DB00F1, 0x100),  # 29-bit UDS physical addressing
                                      *[0x18DB33F1],  # 29-bit UDS functional address
                                      ] for bus in range(4)]

  def setUp(self):
    self.safety = libpanda_py.libpanda
    self.safety.set_safety_hooks(Panda.SAFETY_ELM327, 0)
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

  def test_tx_hook_on_wrong_safety_mode(self):
    # No point, since we allow many diagnostic addresses
    pass


if __name__ == "__main__":
  unittest.main()
