#!/usr/bin/env python3
import unittest

import panda.tests.safety.common as common

from panda import Panda
from panda.tests.libpanda import libpanda_py
from panda.tests.safety.common import CANPackerPanda


class TestBody(common.PandaSafetyTest):
  TX_MSGS = [[0x250, 0], [0x251, 0], [0x350, 0], [0x351, 0],
             [0x1, 0], [0x1, 1], [0x1, 2], [0x1, 3]]

  def setUp(self):
    self.packer = CANPackerPanda("comma_body")
    self.safety = libpanda_py.libpanda
    self.safety.set_safety_hooks(Panda.SAFETY_BODY, 0)
    self.safety.init_tests()

  def _motors_data_msg(self, speed_l, speed_r):
    values = {"SPEED_L": speed_l, "SPEED_R": speed_r}
    return self.packer.make_can_msg_panda("MOTORS_DATA", 0, values)

  def _torque_cmd_msg(self, torque_l, torque_r):
    values = {"TORQUE_L": torque_l, "TORQUE_R": torque_r}
    return self.packer.make_can_msg_panda("TORQUE_CMD", 0, values)

  def _knee_torque_cmd_msg(self, torque_l, torque_r):
    values = {"TORQUE_L": torque_l, "TORQUE_R": torque_r}
    return self.packer.make_can_msg_panda("KNEE_TORQUE_CMD", 0, values)

  def _max_motor_rpm_cmd_msg(self, max_rpm_l, max_rpm_r):
    values = {"MAX_RPM_L": max_rpm_l, "MAX_RPM_R": max_rpm_r}
    return self.packer.make_can_msg_panda("MAX_MOTOR_RPM_CMD", 0, values)

  def test_rx_hook(self):
    self.assertFalse(self.safety.get_controls_allowed())
    self.assertFalse(self.safety.get_vehicle_moving())

    # controls allowed when we get MOTORS_DATA message
    self.assertTrue(self._rx(self._torque_cmd_msg(0, 0)))
    self.assertTrue(self.safety.get_vehicle_moving())  # always moving
    self.assertFalse(self.safety.get_controls_allowed())

    self.assertTrue(self._rx(self._motors_data_msg(0, 0)))
    self.assertTrue(self.safety.get_vehicle_moving())  # always moving
    self.assertTrue(self.safety.get_controls_allowed())

  def test_tx_hook(self):
    self.assertFalse(self._tx(self._torque_cmd_msg(0, 0)))
    self.assertFalse(self._tx(self._knee_torque_cmd_msg(0, 0)))
    self.safety.set_controls_allowed(True)
    self.assertTrue(self._tx(self._torque_cmd_msg(0, 0)))
    self.assertTrue(self._tx(self._knee_torque_cmd_msg(0, 0)))

  def test_can_flasher(self):
    # CAN flasher always allowed
    self.safety.set_controls_allowed(False)
    self.assertTrue(self._tx(common.make_msg(0, 0x1, 8)))

    # 0xdeadfaceU enters CAN flashing mode for base & knee
    for addr in (0x250, 0x350):
      self.assertTrue(self._tx(common.make_msg(0, addr, dat=b'\xce\xfa\xad\xde\x1e\x0b\xb0\x0a')))
      self.assertFalse(self._tx(common.make_msg(0, addr, dat=b'\xce\xfa\xad\xde\x1e\x0b\xb0')))  # not correct data/len
      self.assertFalse(self._tx(common.make_msg(0, addr + 1, dat=b'\xce\xfa\xad\xde\x1e\x0b\xb0\x0a')))  # wrong address


if __name__ == "__main__":
  unittest.main()
