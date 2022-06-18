#!/usr/bin/env python3

import unittest

from opendbc.can.parser import CANParser, CANDefine
from opendbc.can.packer import CANPacker

class TestCanParserPackerExceptions(unittest.TestCase):
  def test_civic_exceptions(self):
    dbc_file = "honda_civic_touring_2016_can_generated"
    dbc_invalid = dbc_file + "abcdef"
    signals = [
      ("STEER_TORQUE", "STEERING_CONTROL"),
      ("STEER_TORQUE_REQUEST", "STEERING_CONTROL"),
    ]
    checks = [("STEERING_CONTROL", 50)]
    with self.assertRaises(RuntimeError):
      CANParser(dbc_invalid, signals, checks, 0)
    with self.assertRaises(RuntimeError):
      CANPacker(dbc_invalid)
    with self.assertRaises(RuntimeError):
      CANDefine(dbc_invalid)

    with self.assertRaises(RuntimeError):
      CANParser(dbc_file, signals, [], 0)

    # Everything is supposed to work below
    CANParser(dbc_file, signals, checks, 0)
    CANPacker(dbc_file)
    CANDefine(dbc_file)


if __name__ == "__main__":
  unittest.main()
