#!/usr/bin/env python3
import glob
import os
import unittest

from opendbc import DBC_PATH
from opendbc.can.parser import CANParser


class TestDBCParser(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.dbcs = []
    for dbc in glob.glob(f"{DBC_PATH}/*.dbc"):
      cls.dbcs.append(os.path.basename(dbc).split('.')[0])

  def test_parse_all_dbcs(self):
    """
      Dynamic DBC parser checks:
        - Checksum and counter length, start bit, endianness
        - Duplicate message addresses and names
        - Signal out of bounds
        - All BO_, SG_, VAL_ lines for syntax errors
    """

    for dbc in self.dbcs:
      CANParser(dbc, [], [], 0)


if __name__ == "__main__":
  unittest.main()
