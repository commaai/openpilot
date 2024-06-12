#!/usr/bin/env python3
import unittest

from opendbc.can.can_define import CANDefine
from opendbc.can.tests import ALL_DBCS


class TestCADNDefine(unittest.TestCase):
  def test_civic(self):

    dbc_file = "honda_civic_touring_2016_can_generated"
    defs = CANDefine(dbc_file)

    self.assertDictEqual(defs.dv[399], defs.dv['STEER_STATUS'])
    self.assertDictEqual(defs.dv[399],
                         {'STEER_STATUS':
                          {7: 'PERMANENT_FAULT',
                           6: 'TMP_FAULT',
                           5: 'FAULT_1',
                           4: 'NO_TORQUE_ALERT_2',
                           3: 'LOW_SPEED_LOCKOUT',
                           2: 'NO_TORQUE_ALERT_1',
                           0: 'NORMAL'}
                          }
                         )

  def test_all_dbcs(self):
    # Asserts no exceptions on all DBCs
    for dbc in ALL_DBCS:
      with self.subTest(dbc=dbc):
        CANDefine(dbc)


if __name__ == "__main__":
  unittest.main()
