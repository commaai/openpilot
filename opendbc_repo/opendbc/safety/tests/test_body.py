#!/usr/bin/env python3
import unittest

from opendbc.car.structs import CarParams
import opendbc.safety.tests.common as common
from opendbc.safety.tests.libsafety import libsafety_py
from opendbc.safety.tests.common import CANPackerSafety
from opendbc.car.body.bodycan import body_checksum


def checksum(msg):
  addr, dat, bus = msg
  dat = bytearray(dat)
  dat[-1] = body_checksum(addr, None, dat)
  return addr, bytes(dat), bus


class TestBody(common.SafetyTest):
  TX_MSGS = [[0x250, 0], [0x251, 0],
             [0x1, 0], [0x1, 1], [0x1, 2], [0x1, 3]]
  FWD_BUS_LOOKUP = {}
  counter = 0

  def setUp(self):
    self.packer = CANPackerSafety("comma_body")
    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(CarParams.SafetyModel.body, 0)
    self.safety.init_tests()

  def _motors_data_msg(self, speed_l, speed_r):
    values = {"SPEED_L": speed_l, "SPEED_R": speed_r, "COUNTER": self.counter % 16}
    self.__class__.counter += 1
    return self.packer.make_can_msg_safety("MOTORS_DATA", 0, values, fix_checksum=checksum)

  def _torque_cmd_msg(self, torque_l, torque_r):
    values = {"TORQUE_L": torque_l, "TORQUE_R": torque_r}
    return self.packer.make_can_msg_safety("TORQUE_CMD", 0, values)

  def test_rx_hook(self):
    self.assertFalse(self.safety.get_controls_allowed())

    # controls allowed when we get MOTORS_DATA message
    self.assertTrue(self._rx(self._torque_cmd_msg(0, 0)))
    self.assertFalse(self.safety.get_controls_allowed())

    self.assertTrue(self._rx(self._motors_data_msg(0, 0)))
    self.assertTrue(self.safety.get_controls_allowed())

    self._reset_safety_hooks()
    for _ in range(16):
      self.assertTrue(self._rx(self._motors_data_msg(0, 0)))

    msg = self._motors_data_msg(0, 0)
    msg[0].data[7] ^= 0xff
    self.assertFalse(self._rx(msg))

    self._reset_safety_hooks()
    for _ in range(16):
      self.assertTrue(self._rx(self._motors_data_msg(0, 0)))

    msg = self._motors_data_msg(0, 0)
    for _ in range(common.MAX_WRONG_COUNTERS + 1):
      valid = self._rx(msg)
    self.assertFalse(valid)

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
    self.assertFalse(self._tx(common.make_msg(0, 0x250, dat=b'\xcf\xfa\xad\xde\x1e\x0b\xb0\x0a')))  # wrong signature
    self.assertFalse(self._tx(common.make_msg(0, 0x250, dat=b'\xce\xfa\xad\xde\x1e\x0b\xb0')))  # not correct data/len
    self.assertFalse(self._tx(common.make_msg(0, 0x251, dat=b'\xce\xfa\xad\xde\x1e\x0b\xb0\x0a')))  # wrong address


if __name__ == "__main__":
  unittest.main()
