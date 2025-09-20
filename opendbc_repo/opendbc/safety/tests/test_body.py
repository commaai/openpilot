#!/usr/bin/env python3
import unittest

from opendbc.car.structs import CarParams
import opendbc.safety.tests.common as common
from opendbc.safety.tests.libsafety import libsafety_py
from opendbc.safety.tests.common import CANPackerPanda


class TestBody(common.PandaSafetyTest):
  TX_MSGS = [[0x250, 0], [0x251, 0],
             [0x1, 0], [0x1, 1], [0x1, 2], [0x1, 3]]
  FWD_BUS_LOOKUP = {}

  def setUp(self):
    self.packer = CANPackerPanda("comma_body")
    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(CarParams.SafetyModel.body, 0)
    self.safety.init_tests()

  def _motors_data_msg(self, speed_l, speed_r):
    values = {"SPEED_L": speed_l, "SPEED_R": speed_r}
    return self.packer.make_can_msg_panda("MOTORS_DATA", 0, values)

  def _torque_cmd_msg(self, torque_l, torque_r):
    values = {"TORQUE_L": torque_l, "TORQUE_R": torque_r}
    return self.packer.make_can_msg_panda("TORQUE_CMD", 0, values)

  def _max_motor_rpm_cmd_msg(self, max_rpm_l, max_rpm_r):
    values = {"MAX_RPM_L": max_rpm_l, "MAX_RPM_R": max_rpm_r}
    return self.packer.make_can_msg_panda("MAX_MOTOR_RPM_CMD", 0, values)

  def test_rx_hook(self):
    self.assertFalse(self.safety.get_controls_allowed())

    # controls allowed when we get MOTORS_DATA message
    self.assertTrue(self._rx(self._torque_cmd_msg(0, 0)))
    self.assertFalse(self.safety.get_controls_allowed())

    self.assertTrue(self._rx(self._motors_data_msg(0, 0)))
    self.assertTrue(self.safety.get_controls_allowed())

  def test_tx_hook(self):
    self.assertFalse(self._tx(self._torque_cmd_msg(0, 0)))
    self.safety.set_controls_allowed(True)
    self.assertTrue(self._tx(self._torque_cmd_msg(0, 0)))

  def test_can_flasher(self):
    # CAN flasher always allowed
    self.safety.set_controls_allowed(False)
    self.assertTrue(self._tx(common.make_msg(0, 0x1, 8)))

    # 0xdeadfaceU allowed for CAN flashing mode
    self.assertTrue(self._tx(common.make_msg(0, 0x250, dat=b'\xce\xfa\xad\xde\x1e\x0b\xb0\x0a')))
    self.assertFalse(self._tx(common.make_msg(0, 0x250, dat=b'\xce\xfa\xad\xde\x1e\x0b\xb0')))  # not correct data/len
    self.assertFalse(self._tx(common.make_msg(0, 0x251, dat=b'\xce\xfa\xad\xde\x1e\x0b\xb0\x0a')))  # wrong address


if __name__ == "__main__":
  unittest.main()
