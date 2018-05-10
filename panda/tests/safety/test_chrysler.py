#!/usr/bin/env python2
import unittest
import numpy as np
import libpandasafety_py


class TestChryslerSafety(unittest.TestCase):
  @classmethod
  def setUp(cls):
    cls.safety = libpandasafety_py.libpandasafety
    cls.safety.chrysler_init(0)
    cls.safety.init_tests_chrysler()

  def _brake_msg(self, brake):
    to_send = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
    to_send[0].RIR = 320 << 21
    to_send[0].RDLR = 0x485 if brake else 0x480
    return to_send

  def _steer_msg(self, angle):
    to_send = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
    to_send[0].RIR = 658 << 21
    c_angle = (1024 + angle)
    to_send[0].RDLR = 0x10 | ((c_angle & 0xf00) >> 8) | ((c_angle & 0xff) << 8)
    return to_send

  
  def test_default_controls_not_allowed(self):
    self.assertFalse(self.safety.get_controls_allowed())

  def test_disengage_on_brake(self):
    self.safety.set_controls_allowed(1)
    self.safety.chrysler_rx_hook(self._brake_msg(1))
    self.assertFalse(self.safety.get_controls_allowed())

  def test_steer_calc(self):
    self.assertEqual(0x14, self._steer_msg(0)[0].RDLR)  # straight, no steering

  def test_steer_tx(self):
    self.assertTrue(self.safety.chrysler_tx_hook(self._steer_msg(0)))
    self.assertTrue(self.safety.chrysler_tx_hook(self._steer_msg(100)))
    self.assertTrue(self.safety.chrysler_tx_hook(self._steer_msg(-100)))
    self.assertFalse(self.safety.chrysler_tx_hook(self._steer_msg(300)))
    self.assertFalse(self.safety.chrysler_tx_hook(self._steer_msg(-300)))

if __name__ == "__main__":
  unittest.main()
