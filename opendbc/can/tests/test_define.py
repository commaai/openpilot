#!/usr/bin/env python3
import unittest

from opendbc.can.can_define import CANDefine


class TestCADNDefine(unittest.TestCase):
  def test_civic(self):

    dbc_file = "honda_civic_touring_2016_can_generated"
    defs = CANDefine(dbc_file)

    self.assertDictEqual(defs.dv[399], defs.dv['STEER_STATUS'])
    self.assertDictEqual(defs.dv[399],
                         {'STEER_STATUS':
                          {6: 'TMP_FAULT',
                           5: 'FAULT_1',
                           4: 'NO_TORQUE_ALERT_2',
                           3: 'LOW_SPEED_LOCKOUT',
                           2: 'NO_TORQUE_ALERT_1',
                           0: 'NORMAL'}
                          }
                         )


if __name__ == "__main__":
  unittest.main()
