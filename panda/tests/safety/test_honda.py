#!/usr/bin/env python3
import unittest
import numpy as np
from panda import Panda
from panda.tests.safety import libpandasafety_py
from panda.tests.safety.common import test_relay_malfunction, make_msg, \
                                      test_manually_enable_controls_allowed, \
                                      test_spam_can_buses, MAX_WRONG_COUNTERS

MAX_BRAKE = 255

INTERCEPTOR_THRESHOLD = 328
TX_MSGS = [[0xE4, 0], [0x194, 0], [0x1FA, 0], [0x200, 0], [0x30C, 0], [0x33D, 0]]

HONDA_N_HW = 0
HONDA_BG_HW = 1
HONDA_BH_HW = 2

def honda_checksum(msg, addr, len_msg):
  checksum = 0
  while addr > 0:
    checksum += addr
    addr >>= 4
  for i in range (0, 2*len_msg):
    if i < 8:
      checksum += (msg.RDLR >> (4 * i))
    else:
      checksum += (msg.RDHR >> (4 * (i - 8)))
  return (8 - checksum) & 0xF


class TestHondaSafety(unittest.TestCase):
  @classmethod
  def setUp(cls):
    cls.safety = libpandasafety_py.libpandasafety
    cls.safety.set_safety_hooks(Panda.SAFETY_HONDA_NIDEC, 0)
    cls.safety.init_tests_honda()
    cls.cnt_speed = 0
    cls.cnt_gas = 0
    cls.cnt_button = 0

  def _speed_msg(self, speed):
    to_send = make_msg(0, 0x158)
    to_send[0].RDLR = speed
    to_send[0].RDHR |= (self.cnt_speed % 4) << 28
    to_send[0].RDHR |= honda_checksum(to_send[0], 0x158, 8) << 24
    self.cnt_speed += 1
    return to_send

  def _button_msg(self, buttons, addr):
    honda_hw = self.safety.get_honda_hw()
    bus = 1 if honda_hw == HONDA_BH_HW else 0
    to_send = make_msg(bus, addr)
    to_send[0].RDLR = buttons << 5
    to_send[0].RDHR |= (self.cnt_button % 4) << 28
    to_send[0].RDHR |= honda_checksum(to_send[0], addr, 8) << 24
    self.cnt_button += 1
    return to_send

  def _brake_msg(self, brake):
    to_send = make_msg(0, 0x17C)
    to_send[0].RDHR = 0x200000 if brake else 0
    to_send[0].RDHR |= (self.cnt_gas % 4) << 28
    to_send[0].RDHR |= honda_checksum(to_send[0], 0x17C, 8) << 24
    self.cnt_gas += 1
    return to_send

  def _alt_brake_msg(self, brake):
    to_send = make_msg(0, 0x1BE)
    to_send[0].RDLR = 0x10 if brake else 0
    return to_send

  def _gas_msg(self, gas):
    to_send = make_msg(0, 0x17C)
    to_send[0].RDLR = 1 if gas else 0
    to_send[0].RDHR |= (self.cnt_gas % 4) << 28
    to_send[0].RDHR |= honda_checksum(to_send[0], 0x17C, 8) << 24
    self.cnt_gas += 1
    return to_send

  def _send_brake_msg(self, brake):
    to_send = make_msg(0, 0x1FA)
    to_send[0].RDLR = ((brake & 0x3) << 14) | ((brake & 0x3FF) >> 2)
    return to_send

  def _send_interceptor_msg(self, gas, addr):
    to_send = make_msg(0, addr, 6)
    gas2 = gas * 2
    to_send[0].RDLR = ((gas & 0xff) << 8) | ((gas & 0xff00) >> 8) | \
                      ((gas2 & 0xff) << 24) | ((gas2 & 0xff00) << 8)
    return to_send

  def _send_steer_msg(self, steer):
    to_send = make_msg(0, 0xE4, 6)
    to_send[0].RDLR = steer
    return to_send

  def test_spam_can_buses(self):
    self.safety.set_honda_hw(HONDA_N_HW)
    test_spam_can_buses(self, TX_MSGS)

  def test_relay_malfunction(self):
    test_relay_malfunction(self, 0xE4)

  def test_default_controls_not_allowed(self):
    self.assertFalse(self.safety.get_controls_allowed())

  def test_manually_enable_controls_allowed(self):
    test_manually_enable_controls_allowed(self)

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
    self.assertEqual(0, self.safety.get_honda_moving())
    self.safety.safety_rx_hook(self._speed_msg(100))
    self.assertEqual(1, self.safety.get_honda_moving())

  def test_prev_brake(self):
    self.assertFalse(self.safety.get_honda_brake_pressed_prev())
    self.safety.safety_rx_hook(self._brake_msg(True))
    self.assertTrue(self.safety.get_honda_brake_pressed_prev())

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
    self.safety.safety_rx_hook(self._gas_msg(0))
    self.safety.set_controls_allowed(1)
    self.safety.safety_rx_hook(self._gas_msg(1))
    self.assertFalse(self.safety.get_controls_allowed())

  def test_allow_engage_with_gas_pressed(self):
    self.safety.safety_rx_hook(self._gas_msg(1))
    self.safety.set_controls_allowed(1)
    self.safety.safety_rx_hook(self._gas_msg(1))
    self.assertTrue(self.safety.get_controls_allowed())

  def test_disengage_on_gas_interceptor(self):
    for g in range(0, 0x1000):
      self.safety.safety_rx_hook(self._send_interceptor_msg(0, 0x201))
      self.safety.set_controls_allowed(True)
      self.safety.safety_rx_hook(self._send_interceptor_msg(g, 0x201))
      remain_enabled = g <= INTERCEPTOR_THRESHOLD
      self.assertEqual(remain_enabled, self.safety.get_controls_allowed())
      self.safety.safety_rx_hook(self._send_interceptor_msg(0, 0x201))
      self.safety.set_gas_interceptor_detected(False)

  def test_allow_engage_with_gas_interceptor_pressed(self):
    self.safety.safety_rx_hook(self._send_interceptor_msg(0x1000, 0x201))
    self.safety.set_controls_allowed(1)
    self.safety.safety_rx_hook(self._send_interceptor_msg(0x1000, 0x201))
    self.assertTrue(self.safety.get_controls_allowed())
    self.safety.safety_rx_hook(self._send_interceptor_msg(0, 0x201))
    self.safety.set_gas_interceptor_detected(False)

  def test_brake_safety_check(self):
    for fwd_brake in [False, True]:
      self.safety.set_honda_fwd_brake(fwd_brake)
      for brake in np.arange(0, MAX_BRAKE + 10, 1):
        for controls_allowed in [True, False]:
          self.safety.set_controls_allowed(controls_allowed)
          if fwd_brake:
            send = False  # block openpilot brake msg when fwd'ing stock msg
          elif controls_allowed:
            send = MAX_BRAKE >= brake >= 0
          else:
            send = brake == 0
          self.assertEqual(send, self.safety.safety_tx_hook(self._send_brake_msg(brake)))
    self.safety.set_honda_fwd_brake(False)

  def test_gas_interceptor_safety_check(self):
    for gas in np.arange(0, 4000, 100):
      for controls_allowed in [True, False]:
        self.safety.set_controls_allowed(controls_allowed)
        if controls_allowed:
          send = True
        else:
          send = gas == 0
        self.assertEqual(send, self.safety.safety_tx_hook(self._send_interceptor_msg(gas, 0x200)))

  def test_steer_safety_check(self):
    self.safety.set_controls_allowed(0)
    self.assertTrue(self.safety.safety_tx_hook(self._send_steer_msg(0x0000)))
    self.assertFalse(self.safety.safety_tx_hook(self._send_steer_msg(0x1000)))

  def test_spam_cancel_safety_check(self):
    RESUME_BTN = 4
    SET_BTN = 3
    CANCEL_BTN = 2
    BUTTON_MSG = 0x296
    for hw in [HONDA_BG_HW, HONDA_BH_HW]:
      self.safety.set_honda_hw(hw)
      self.safety.set_controls_allowed(0)
      self.assertTrue(self.safety.safety_tx_hook(self._button_msg(CANCEL_BTN, BUTTON_MSG)))
      self.assertFalse(self.safety.safety_tx_hook(self._button_msg(RESUME_BTN, BUTTON_MSG)))
      self.assertFalse(self.safety.safety_tx_hook(self._button_msg(SET_BTN, BUTTON_MSG)))
      # do not block resume if we are engaged already
      self.safety.set_controls_allowed(1)
      self.assertTrue(self.safety.safety_tx_hook(self._button_msg(RESUME_BTN, BUTTON_MSG)))

  def test_rx_hook(self):
    # checksum checks
    SET_BTN = 3
    for msg in ["btn1", "btn2", "gas", "speed"]:
      self.safety.set_controls_allowed(1)
      if msg == "btn1":
        to_push = self._button_msg(SET_BTN, 0x1A6)
      if msg == "btn2":
        to_push = self._button_msg(SET_BTN, 0x296)
      if msg == "gas":
        to_push = self._gas_msg(0)
      if msg == "speed":
        to_push = self._speed_msg(0)
      self.assertTrue(self.safety.safety_rx_hook(to_push))
      to_push[0].RDHR = 0
      self.assertFalse(self.safety.safety_rx_hook(to_push))
      self.assertFalse(self.safety.get_controls_allowed())

    # counter
    # reset wrong_counters to zero by sending valid messages
    for i in range(MAX_WRONG_COUNTERS + 1):
      self.cnt_speed = 0
      self.cnt_gas = 0
      self.cnt_button = 0
      if i < MAX_WRONG_COUNTERS:
        self.safety.set_controls_allowed(1)
        self.safety.safety_rx_hook(self._button_msg(SET_BTN, 0x1A6))
        self.safety.safety_rx_hook(self._speed_msg(0))
        self.safety.safety_rx_hook(self._gas_msg(0))
      else:
        self.assertFalse(self.safety.safety_rx_hook(self._button_msg(SET_BTN, 0x1A6)))
        self.assertFalse(self.safety.safety_rx_hook(self._speed_msg(0)))
        self.assertFalse(self.safety.safety_rx_hook(self._gas_msg(0)))
        self.assertFalse(self.safety.get_controls_allowed())

    # restore counters for future tests with a couple of good messages
    for i in range(2):
      self.safety.set_controls_allowed(1)
      self.safety.safety_rx_hook(self._button_msg(SET_BTN, 0x1A6))
      self.safety.safety_rx_hook(self._speed_msg(0))
      self.safety.safety_rx_hook(self._gas_msg(0))
    self.safety.safety_rx_hook(self._button_msg(SET_BTN, 0x1A6))
    self.assertTrue(self.safety.get_controls_allowed())


  def test_fwd_hook(self):
    buss = list(range(0x0, 0x3))
    msgs = list(range(0x1, 0x800))
    fwd_brake = [False, True]

    self.safety.set_honda_hw(HONDA_N_HW)

    for f in fwd_brake:
      self.safety.set_honda_fwd_brake(f)
      blocked_msgs = [0xE4, 0x194, 0x33D]
      blocked_msgs += [0x30C]
      if not f:
        blocked_msgs += [0x1FA]
      for b in buss:
        for m in msgs:
          if b == 0:
            fwd_bus = 2
          elif b == 1:
            fwd_bus = -1
          elif b == 2:
            fwd_bus = -1 if m in blocked_msgs else 0

          # assume len 8
          self.assertEqual(fwd_bus, self.safety.safety_fwd_hook(b, make_msg(b, m, 8)))

    self.safety.set_honda_fwd_brake(False)


if __name__ == "__main__":
  unittest.main()
