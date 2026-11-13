import unittest

from opendbc.can import CANDefine, CANPacker, CANParser
from opendbc.can.tests import TEST_DBC


class TestCanParserPackerExceptions(unittest.TestCase):
  def test_civic_exceptions(self):
    dbc_file = "honda_civic_touring_2016_can_generated"
    dbc_invalid = dbc_file + "abcdef"
    msgs = [("STEERING_CONTROL", 50)]
    with self.assertRaises(FileNotFoundError):
      CANParser(dbc_invalid, msgs, 0)
    with self.assertRaises(FileNotFoundError):
      CANPacker(dbc_invalid)
    with self.assertRaises(FileNotFoundError):
      CANDefine(dbc_invalid)
    with self.assertRaises(KeyError):
      CANDefine(TEST_DBC)

    parser = CANParser(dbc_file, msgs, 0)
    with self.assertRaises(IndexError):
      parser.update([b''])

    # Everything is supposed to work below
    CANParser(dbc_file, msgs, 0)
    CANParser(dbc_file, [], 0)
    CANPacker(dbc_file)
    CANDefine(dbc_file)
