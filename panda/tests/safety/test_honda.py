#!/usr/bin/env python2
import unittest
import numpy as np
import libpandasafety_py

MAX_BRAKE = 255

class TestHondaSafety(unittest.TestCase):
  @classmethod
  def setUp(cls):
    cls.safety = libpandasafety_py.libpandasafety
    cls.safety.safety_set_mode(1, 0)
    cls.safety.init_tests_honda()

  def _send_msg(self, bus, addr, length):
    to_send = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
    to_send[0].RIR = addr << 21
    to_send[0].RDTR = length
    to_send[0].RDTR = bus << 4

    return to_send

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
    to_send[0].RDLR = ((brake & 0x3) << 8) | ((brake & 0x3FF) >> 2)

    return to_send

  def _send_interceptor_msg(self, gas, addr):
    to_send = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
    to_send[0].RIR = addr << 21
    to_send[0].RDTR = 6
    to_send[0].RDLR = ((gas & 0xff) << 8) | ((gas & 0xff00) >> 8)

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
    self.safety.safety_rx_hook(self._button_msg(RESUME_BTN, 0x1A6))
    self.assertTrue(self.safety.get_controls_allowed())

  def test_set_button(self):
    SET_BTN = 3
    self.safety.set_controls_allowed(0)
    self.safety.safety_rx_hook(self._button_msg(SET_BTN, 0x1A6))
    self.assertTrue(self.safety.get_controls_allowed())

  def test_cancel_button(self):
    CANCEL_BTN = 2
    self.safety.set_controls_allowed(1)
    self.safety.safety_rx_hook(self._button_msg(CANCEL_BTN, 0x1A6))
    self.assertFalse(self.safety.get_controls_allowed())

  def test_sample_speed(self):
    self.assertEqual(0, self.safety.get_honda_ego_speed())
    self.safety.safety_rx_hook(self._speed_msg(100))
    self.assertEqual(100, self.safety.get_honda_ego_speed())

  def test_prev_brake(self):
    self.assertFalse(self.safety.get_honda_brake_prev())
    self.safety.safety_rx_hook(self._brake_msg(True))
    self.assertTrue(self.safety.get_honda_brake_prev())

  def test_disengage_on_brake(self):
    self.safety.set_controls_allowed(1)
    self.safety.safety_rx_hook(self._brake_msg(1))
    self.assertFalse(self.safety.get_controls_allowed())

  def test_alt_disengage_on_brake(self):
    self.safety.set_honda_alt_brake_msg(1)
    self.safety.set_controls_allowed(1)
    self.safety.safety_rx_hook(self._alt_brake_msg(1))
    self.assertFalse(self.safety.get_controls_allowed())

    self.safety.set_honda_alt_brake_msg(0)
    self.safety.set_controls_allowed(1)
    self.safety.safety_rx_hook(self._alt_brake_msg(1))
    self.assertTrue(self.safety.get_controls_allowed())

  def test_allow_brake_at_zero_speed(self):
    # Brake was already pressed
    self.safety.safety_rx_hook(self._brake_msg(True))
    self.safety.set_controls_allowed(1)

    self.safety.safety_rx_hook(self._brake_msg(True))
    self.assertTrue(self.safety.get_controls_allowed())
    self.safety.safety_rx_hook(self._brake_msg(False))  # reset no brakes

  def test_not_allow_brake_when_moving(self):
    # Brake was already pressed
    self.safety.safety_rx_hook(self._brake_msg(True))
    self.safety.safety_rx_hook(self._speed_msg(100))
    self.safety.set_controls_allowed(1)

    self.safety.safety_rx_hook(self._brake_msg(True))
    self.assertFalse(self.safety.get_controls_allowed())

  def test_prev_gas(self):
    self.safety.safety_rx_hook(self._gas_msg(False))
    self.assertFalse(self.safety.get_honda_gas_prev())
    self.safety.safety_rx_hook(self._gas_msg(True))
    self.assertTrue(self.safety.get_honda_gas_prev())

  def test_prev_gas_interceptor(self):
    self.safety.safety_rx_hook(self._send_interceptor_msg(0x0, 0x201))
    self.assertFalse(self.safety.get_gas_interceptor_prev())
    self.safety.safety_rx_hook(self._send_interceptor_msg(0x1000, 0x201))
    self.assertTrue(self.safety.get_gas_interceptor_prev())
    self.safety.safety_rx_hook(self._send_interceptor_msg(0x0, 0x201))
    self.safety.set_gas_interceptor_detected(False)

  def test_disengage_on_gas(self):
    for long_controls_allowed in [0, 1]:
      self.safety.set_long_controls_allowed(long_controls_allowed)
      self.safety.safety_rx_hook(self._gas_msg(0))
      self.safety.set_controls_allowed(1)
      self.safety.safety_rx_hook(self._gas_msg(1))
      if long_controls_allowed:
        self.assertFalse(self.safety.get_controls_allowed())
      else:
        self.assertTrue(self.safety.get_controls_allowed())
    self.safety.set_long_controls_allowed(True)

  def test_allow_engage_with_gas_pressed(self):
    self.safety.safety_rx_hook(self._gas_msg(1))
    self.safety.set_controls_allowed(1)
    self.safety.safety_rx_hook(self._gas_msg(1))
    self.assertTrue(self.safety.get_controls_allowed())

  def test_disengage_on_gas_interceptor(self):
    for long_controls_allowed in [0, 1]:
      self.safety.set_long_controls_allowed(long_controls_allowed)
      self.safety.safety_rx_hook(self._send_interceptor_msg(0, 0x201))
      self.safety.set_controls_allowed(1)
      self.safety.safety_rx_hook(self._send_interceptor_msg(0x1000, 0x201))
      if long_controls_allowed:
        self.assertFalse(self.safety.get_controls_allowed())
      else:
        self.assertTrue(self.safety.get_controls_allowed())
      self.safety.safety_rx_hook(self._send_interceptor_msg(0, 0x201))
      self.safety.set_gas_interceptor_detected(False)
    self.safety.set_long_controls_allowed(True)

  def test_allow_engage_with_gas_interceptor_pressed(self):
    self.safety.safety_rx_hook(self._send_interceptor_msg(0x1000, 0x201))
    self.safety.set_controls_allowed(1)
    self.safety.safety_rx_hook(self._send_interceptor_msg(0x1000, 0x201))
    self.assertTrue(self.safety.get_controls_allowed())
    self.safety.safety_rx_hook(self._send_interceptor_msg(0, 0x201))
    self.safety.set_gas_interceptor_detected(False)

  def test_brake_safety_check(self):
    for long_controls_allowed in [0, 1]:
      self.safety.set_long_controls_allowed(long_controls_allowed)
      for brake in np.arange(0, MAX_BRAKE + 10, 1):
        for controls_allowed in [True, False]:
          self.safety.set_controls_allowed(controls_allowed)
          if controls_allowed and long_controls_allowed:
            send = MAX_BRAKE >= brake >= 0
          else:
            send = brake == 0
          self.assertEqual(send, self.safety.safety_tx_hook(self._send_brake_msg(brake)))
    self.safety.set_long_controls_allowed(True)

  def test_gas_interceptor_safety_check(self):
    for long_controls_allowed in [0, 1]:
      self.safety.set_long_controls_allowed(long_controls_allowed)
      for gas in np.arange(0, 4000, 100):
        for controls_allowed in [True, False]:
          self.safety.set_controls_allowed(controls_allowed)
          if controls_allowed and long_controls_allowed:
            send = True
          else:
            send = gas == 0
          self.assertEqual(send, self.safety.safety_tx_hook(self._send_interceptor_msg(gas, 0x200)))
    self.safety.set_long_controls_allowed(True)

  def test_steer_safety_check(self):
    self.safety.set_controls_allowed(0)
    self.assertTrue(self.safety.safety_tx_hook(self._send_steer_msg(0x0000)))
    self.assertFalse(self.safety.safety_tx_hook(self._send_steer_msg(0x1000)))

  def test_spam_cancel_safety_check(self):
    RESUME_BTN = 4
    SET_BTN = 3
    CANCEL_BTN = 2
    BUTTON_MSG = 0x296
    self.safety.set_honda_bosch_hardware(1)
    self.safety.set_controls_allowed(0)
    self.assertTrue(self.safety.safety_tx_hook(self._button_msg(CANCEL_BTN, BUTTON_MSG)))
    self.assertFalse(self.safety.safety_tx_hook(self._button_msg(RESUME_BTN, BUTTON_MSG)))
    self.assertFalse(self.safety.safety_tx_hook(self._button_msg(SET_BTN, BUTTON_MSG)))
    # do not block resume if we are engaged already
    self.safety.set_controls_allowed(1)
    self.assertTrue(self.safety.safety_tx_hook(self._button_msg(RESUME_BTN, BUTTON_MSG)))

  def test_fwd_hook(self):
    buss = range(0x0, 0x3)
    msgs = range(0x1, 0x800)
    long_controls_allowed = [0, 1]

    self.safety.set_honda_bosch_hardware(0)

    for l in long_controls_allowed:
      self.safety.set_long_controls_allowed(l)
      blocked_msgs = [0xE4, 0x194, 0x33D]
      if l:
        blocked_msgs += [0x1FA ,0x30C, 0x39F]
      for b in buss:
        for m in msgs:
          if b == 0:
            fwd_bus = 2
          elif b == 1:
            fwd_bus = -1
          elif b == 2:
            fwd_bus = -1 if m in blocked_msgs else 0

          # assume len 8
          self.assertEqual(fwd_bus, self.safety.safety_fwd_hook(b, self._send_msg(b, m, 8)))

    self.safety.set_long_controls_allowed(True)



if __name__ == "__main__":
  unittest.main()
