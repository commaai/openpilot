#!/usr/bin/env python2
import unittest
import numpy as np
import libpandasafety_py


class TestHondaSafety(unittest.TestCase):
  @classmethod
  def setUp(cls):
    cls.safety = libpandasafety_py.libpandasafety
    cls.safety.honda_init(0)
    cls.safety.init_tests_honda()

  def _speed_msg(self, speed):
    to_send = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
    to_send[0].RIR = 0x158 << 21
    to_send[0].RDLR = speed

    return to_send

  def _button_msg(self, buttons, msg):
    to_send = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
    to_send[0].RIR = msg << 21
    to_send[0].RDLR = buttons << 5

    return to_send

  def _brake_msg(self, brake):
    to_send = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
    to_send[0].RIR = 0x17C << 21
    to_send[0].RDHR = 0x200000 if brake else 0

    return to_send

  def _alt_brake_msg(self, brake):
    to_send = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
    to_send[0].RIR = 0x1BE << 21
    to_send[0].RDLR = 0x10 if brake else 0

    return to_send

  def _gas_msg(self, gas):
    to_send = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
    to_send[0].RIR = 0x17C << 21
    to_send[0].RDLR = 1 if gas else 0

    return to_send

  def _send_brake_msg(self, brake):
    to_send = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
    to_send[0].RIR = 0x1FA << 21
    to_send[0].RDLR = brake

    return to_send

  def _send_gas_msg(self, gas):
    to_send = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
    to_send[0].RIR = 0x200 << 21
    to_send[0].RDLR = gas

    return to_send

  def _send_steer_msg(self, steer):
    to_send = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
    to_send[0].RIR = 0xE4 << 21
    to_send[0].RDLR = steer

    return to_send

  def test_default_controls_not_allowed(self):
    self.assertFalse(self.safety.get_controls_allowed())

  def test_resume_button(self):
    RESUME_BTN = 4
    self.safety.set_controls_allowed(0)
    self.safety.honda_rx_hook(self._button_msg(RESUME_BTN, 0x1A6))
    self.assertTrue(self.safety.get_controls_allowed())

  def test_set_button(self):
    SET_BTN = 3
    self.safety.set_controls_allowed(0)
    self.safety.honda_rx_hook(self._button_msg(SET_BTN, 0x1A6))
    self.assertTrue(self.safety.get_controls_allowed())

  def test_cancel_button(self):
    CANCEL_BTN = 2
    self.safety.set_controls_allowed(1)
    self.safety.honda_rx_hook(self._button_msg(CANCEL_BTN, 0x1A6))
    self.assertFalse(self.safety.get_controls_allowed())

  def test_sample_speed(self):
    self.assertEqual(0, self.safety.get_ego_speed())
    self.safety.honda_rx_hook(self._speed_msg(100))
    self.assertEqual(100, self.safety.get_ego_speed())

  def test_prev_brake(self):
    self.assertFalse(self.safety.get_brake_prev())
    self.safety.honda_rx_hook(self._brake_msg(True))
    self.assertTrue(self.safety.get_brake_prev())

  def test_disengage_on_brake(self):
    self.safety.set_controls_allowed(1)
    self.safety.honda_rx_hook(self._brake_msg(1))
    self.assertFalse(self.safety.get_controls_allowed())

  def test_alt_disengage_on_brake(self):
    self.safety.set_honda_alt_brake_msg(1)
    self.safety.set_controls_allowed(1)
    self.safety.honda_rx_hook(self._alt_brake_msg(1))
    self.assertFalse(self.safety.get_controls_allowed())

    self.safety.set_honda_alt_brake_msg(0)
    self.safety.set_controls_allowed(1)
    self.safety.honda_rx_hook(self._alt_brake_msg(1))
    self.assertTrue(self.safety.get_controls_allowed())

  def test_allow_brake_at_zero_speed(self):
    # Brake was already pressed
    self.safety.honda_rx_hook(self._brake_msg(True))
    self.safety.set_controls_allowed(1)

    self.safety.honda_rx_hook(self._brake_msg(True))
    self.assertTrue(self.safety.get_controls_allowed())
    self.safety.honda_rx_hook(self._brake_msg(False))  # reset no brakes

  def test_not_allow_brake_when_moving(self):
    # Brake was already pressed
    self.safety.honda_rx_hook(self._brake_msg(True))
    self.safety.honda_rx_hook(self._speed_msg(100))
    self.safety.set_controls_allowed(1)

    self.safety.honda_rx_hook(self._brake_msg(True))
    self.assertFalse(self.safety.get_controls_allowed())

  def test_prev_gas(self):
    self.assertFalse(self.safety.get_gas_prev())
    self.safety.honda_rx_hook(self._gas_msg(True))
    self.assertTrue(self.safety.get_gas_prev())

  def test_disengage_on_gas(self):
    self.safety.set_controls_allowed(1)
    self.safety.honda_rx_hook(self._gas_msg(1))
    self.assertFalse(self.safety.get_controls_allowed())

  def test_allow_engage_with_gas_pressed(self):
    self.safety.honda_rx_hook(self._gas_msg(1))
    self.safety.set_controls_allowed(1)
    self.safety.honda_rx_hook(self._gas_msg(1))
    self.assertTrue(self.safety.get_controls_allowed())

  def test_brake_safety_check(self):
    self.assertTrue(self.safety.honda_tx_hook(self._send_brake_msg(0x0000)))
    self.assertFalse(self.safety.honda_tx_hook(self._send_brake_msg(0x1000)))

    self.safety.set_controls_allowed(1)
    self.assertTrue(self.safety.honda_tx_hook(self._send_brake_msg(0x1000)))
    self.assertFalse(self.safety.honda_tx_hook(self._send_brake_msg(0x00F0)))

  def test_gas_safety_check(self):
    self.safety.set_controls_allowed(0)
    self.assertTrue(self.safety.honda_tx_hook(self._send_gas_msg(0x0000)))
    self.assertFalse(self.safety.honda_tx_hook(self._send_gas_msg(0x1000)))

  def test_steer_safety_check(self):
    self.safety.set_controls_allowed(0)
    self.assertTrue(self.safety.honda_tx_hook(self._send_steer_msg(0x0000)))
    self.assertFalse(self.safety.honda_tx_hook(self._send_steer_msg(0x1000)))

  def test_spam_cancel_safety_check(self):
    RESUME_BTN = 4
    SET_BTN = 3
    CANCEL_BTN = 2
    BUTTON_MSG = 0x296
    self.safety.set_bosch_hardware(1)
    self.safety.set_controls_allowed(0)
    self.assertTrue(self.safety.honda_tx_hook(self._button_msg(CANCEL_BTN, BUTTON_MSG)))
    self.assertFalse(self.safety.honda_tx_hook(self._button_msg(RESUME_BTN, BUTTON_MSG)))
    self.assertFalse(self.safety.honda_tx_hook(self._button_msg(SET_BTN, BUTTON_MSG)))
    # do not block resume if we are engaged already
    self.safety.set_controls_allowed(1)
    self.assertTrue(self.safety.honda_tx_hook(self._button_msg(RESUME_BTN, BUTTON_MSG)))


if __name__ == "__main__":
  unittest.main()
