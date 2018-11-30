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

  def _acc_msg(self, active):
    to_send = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
    to_send[0].RIR = 0x1f4 << 21
    to_send[0].RDLR = 0xfe3fff1f if active else 0xfe0fff1f
    return to_send
    

  def _brake_msg(self, brake):
    to_send = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
    to_send[0].RIR = 0x140 << 21
    to_send[0].RDLR = 0x485 if brake else 0x480
    return to_send

  def _steer_msg(self, angle):
    to_send = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
    to_send[0].RIR = 0x292 << 21
    c_angle = (1024 + angle)
    to_send[0].RDLR = 0x10 | ((c_angle & 0xf00) >> 8) | ((c_angle & 0xff) << 8)
    return to_send

  def test_default_controls_not_allowed(self):
    self.assertFalse(self.safety.get_controls_allowed())

  def test_acc_enables_controls(self):
    self.safety.set_controls_allowed(0)
    self.safety.chrysler_rx_hook(self._acc_msg(0))
    self.assertFalse(self.safety.get_controls_allowed())
    self.safety.chrysler_rx_hook(self._acc_msg(1))
    self.assertTrue(self.safety.get_controls_allowed())
    self.safety.chrysler_rx_hook(self._acc_msg(0))
    self.assertFalse(self.safety.get_controls_allowed())

  def test_disengage_on_brake(self):
    self.safety.set_controls_allowed(1)
    self.safety.chrysler_rx_hook(self._brake_msg(0))
    self.assertTrue(self.safety.get_controls_allowed())
    self.safety.chrysler_rx_hook(self._brake_msg(1))
    self.assertFalse(self.safety.get_controls_allowed())

  def test_steer_calculation(self):
    self.assertEqual(0x14, self._steer_msg(0)[0].RDLR)  # straight, no steering

  def test_steer_tx(self):
    self.safety.set_controls_allowed(1)
    self.assertTrue(self.safety.chrysler_tx_hook(self._steer_msg(0)))
    self.safety.set_chrysler_desired_torque_last(227)
    self.assertTrue(self.safety.chrysler_tx_hook(self._steer_msg(230)))
    self.assertFalse(self.safety.chrysler_tx_hook(self._steer_msg(231)))
    self.safety.set_chrysler_desired_torque_last(-227)
    self.assertFalse(self.safety.chrysler_tx_hook(self._steer_msg(-231)))
    self.assertTrue(self.safety.chrysler_tx_hook(self._steer_msg(-230)))
    # verify max change
    self.safety.set_chrysler_desired_torque_last(0)
    self.assertFalse(self.safety.chrysler_tx_hook(self._steer_msg(230)))
    
    self.safety.set_controls_allowed(0)
    self.safety.set_chrysler_desired_torque_last(0)
    self.assertFalse(self.safety.chrysler_tx_hook(self._steer_msg(3)))
    self.assertTrue(self.safety.chrysler_tx_hook(self._steer_msg(0)))
    # verify when controls not allowed we can still go back towards 0
    self.safety.set_chrysler_desired_torque_last(10)
    self.assertFalse(self.safety.chrysler_tx_hook(self._steer_msg(10)))
    self.assertFalse(self.safety.chrysler_tx_hook(self._steer_msg(11)))
    self.assertTrue(self.safety.chrysler_tx_hook(self._steer_msg(7)))
    self.assertTrue(self.safety.chrysler_tx_hook(self._steer_msg(4)))
    self.assertTrue(self.safety.chrysler_tx_hook(self._steer_msg(0)))
    self.safety.set_chrysler_desired_torque_last(-10)
    self.assertFalse(self.safety.chrysler_tx_hook(self._steer_msg(-10)))
    self.assertFalse(self.safety.chrysler_tx_hook(self._steer_msg(-11)))
    self.assertTrue(self.safety.chrysler_tx_hook(self._steer_msg(-7)))
    self.assertTrue(self.safety.chrysler_tx_hook(self._steer_msg(-4)))
    self.assertTrue(self.safety.chrysler_tx_hook(self._steer_msg(0)))

if __name__ == "__main__":
  unittest.main()
