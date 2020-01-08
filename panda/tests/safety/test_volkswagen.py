#!/usr/bin/env python3
import unittest
import numpy as np
from panda import Panda
from panda.tests.safety import libpandasafety_py
from panda.tests.safety.common import test_relay_malfunction, make_msg, test_manually_enable_controls_allowed, test_spam_can_buses

MAX_RATE_UP = 4
MAX_RATE_DOWN = 10
MAX_STEER = 250

MAX_RT_DELTA = 75
RT_INTERVAL = 250000

DRIVER_TORQUE_ALLOWANCE = 80
DRIVER_TORQUE_FACTOR = 3

TX_MSGS = [[0x126, 0], [0x12B, 0], [0x12B, 2], [0x397, 0]]

def sign(a):
  if a > 0:
    return 1
  else:
    return -1

class TestVolkswagenSafety(unittest.TestCase):
  @classmethod
  def setUp(cls):
    cls.safety = libpandasafety_py.libpandasafety
    cls.safety.set_safety_hooks(Panda.SAFETY_VOLKSWAGEN, 0)
    cls.safety.init_tests_volkswagen()

  def _set_prev_torque(self, t):
    self.safety.set_volkswagen_desired_torque_last(t)
    self.safety.set_volkswagen_rt_torque_last(t)

  def _torque_driver_msg(self, torque):
    to_send = make_msg(0, 0x9F)
    t = abs(torque)
    to_send[0].RDHR = ((t & 0x1FFF) << 8)
    if torque < 0:
      to_send[0].RDHR |= 0x1 << 23
    return to_send

  def _torque_msg(self, torque):
    to_send = make_msg(0, 0x126)
    t = abs(torque)
    to_send[0].RDLR = (t & 0xFFF) << 16
    if torque < 0:
      to_send[0].RDLR |= 0x1 << 31
    return to_send

  def _gas_msg(self, gas):
    to_send = make_msg(0, 0x121)
    to_send[0].RDLR = (gas & 0xFF) << 12
    return to_send

  def _button_msg(self, bit):
    to_send = make_msg(2, 0x12B)
    to_send[0].RDLR = 1 << bit
    return to_send

  def test_spam_can_buses(self):
    test_spam_can_buses(self, TX_MSGS)

  def test_relay_malfunction(self):
    test_relay_malfunction(self, 0x126)

  def test_prev_gas(self):
    for g in range(0, 256):
      self.safety.safety_rx_hook(self._gas_msg(g))
      self.assertEqual(g, self.safety.get_volkswagen_gas_prev())

  def test_default_controls_not_allowed(self):
    self.assertFalse(self.safety.get_controls_allowed())

  def test_enable_control_allowed_from_cruise(self):
    to_push = make_msg(0, 0x122)
    to_push[0].RDHR = 0x30000000
    self.safety.safety_rx_hook(to_push)
    self.assertTrue(self.safety.get_controls_allowed())

  def test_disable_control_allowed_from_cruise(self):
    to_push = make_msg(0, 0x122)
    self.safety.set_controls_allowed(1)
    self.safety.safety_rx_hook(to_push)
    self.assertFalse(self.safety.get_controls_allowed())

  def test_disengage_on_gas(self):
    for long_controls_allowed in [0, 1]:
      self.safety.set_long_controls_allowed(long_controls_allowed)
      self.safety.safety_rx_hook(self._gas_msg(0))
      self.safety.set_controls_allowed(True)
      self.safety.safety_rx_hook(self._gas_msg(1))
      if long_controls_allowed:
        self.assertFalse(self.safety.get_controls_allowed())
      else:
        self.assertTrue(self.safety.get_controls_allowed())
    self.safety.set_long_controls_allowed(True)

  def test_allow_engage_with_gas_pressed(self):
    self.safety.safety_rx_hook(self._gas_msg(1))
    self.safety.set_controls_allowed(True)
    self.safety.safety_rx_hook(self._gas_msg(1))
    self.assertTrue(self.safety.get_controls_allowed())
    self.safety.safety_rx_hook(self._gas_msg(1))
    self.assertTrue(self.safety.get_controls_allowed())


  def test_steer_safety_check(self):
    for enabled in [0, 1]:
      for t in range(-500, 500):
        self.safety.set_controls_allowed(enabled)
        self._set_prev_torque(t)
        if abs(t) > MAX_STEER or (not enabled and abs(t) > 0):
          self.assertFalse(self.safety.safety_tx_hook(self._torque_msg(t)))
        else:
          self.assertTrue(self.safety.safety_tx_hook(self._torque_msg(t)))

  def test_manually_enable_controls_allowed(self):
    test_manually_enable_controls_allowed(self)

  def test_spam_cancel_safety_check(self):
    BIT_CANCEL = 13
    BIT_RESUME = 19
    BIT_SET = 16
    self.safety.set_controls_allowed(0)
    self.assertTrue(self.safety.safety_tx_hook(self._button_msg(BIT_CANCEL)))
    self.assertFalse(self.safety.safety_tx_hook(self._button_msg(BIT_RESUME)))
    self.assertFalse(self.safety.safety_tx_hook(self._button_msg(BIT_SET)))
    # do not block resume if we are engaged already
    self.safety.set_controls_allowed(1)
    self.assertTrue(self.safety.safety_tx_hook(self._button_msg(BIT_RESUME)))

  def test_non_realtime_limit_up(self):
    self.safety.set_volkswagen_torque_driver(0, 0)
    self.safety.set_controls_allowed(True)

    self._set_prev_torque(0)
    self.assertTrue(self.safety.safety_tx_hook(self._torque_msg(MAX_RATE_UP)))
    self._set_prev_torque(0)
    self.assertTrue(self.safety.safety_tx_hook(self._torque_msg(-MAX_RATE_UP)))

    self._set_prev_torque(0)
    self.assertFalse(self.safety.safety_tx_hook(self._torque_msg(MAX_RATE_UP + 1)))
    self.safety.set_controls_allowed(True)
    self._set_prev_torque(0)
    self.assertFalse(self.safety.safety_tx_hook(self._torque_msg(-MAX_RATE_UP - 1)))

  def test_non_realtime_limit_down(self):
    self.safety.set_volkswagen_torque_driver(0, 0)
    self.safety.set_controls_allowed(True)

  def test_against_torque_driver(self):
    self.safety.set_controls_allowed(True)

    for sign in [-1, 1]:
      for t in np.arange(0, DRIVER_TORQUE_ALLOWANCE + 1, 1):
        t *= -sign
        self.safety.set_volkswagen_torque_driver(t, t)
        self._set_prev_torque(MAX_STEER * sign)
        self.assertTrue(self.safety.safety_tx_hook(self._torque_msg(MAX_STEER * sign)))

      self.safety.set_volkswagen_torque_driver(DRIVER_TORQUE_ALLOWANCE + 1, DRIVER_TORQUE_ALLOWANCE + 1)
      self.assertFalse(self.safety.safety_tx_hook(self._torque_msg(-MAX_STEER)))

    # spot check some individual cases
    for sign in [-1, 1]:
      driver_torque = (DRIVER_TORQUE_ALLOWANCE + 10) * sign
      torque_desired = (MAX_STEER - 10 * DRIVER_TORQUE_FACTOR) * sign
      delta = 1 * sign
      self._set_prev_torque(torque_desired)
      self.safety.set_volkswagen_torque_driver(-driver_torque, -driver_torque)
      self.assertTrue(self.safety.safety_tx_hook(self._torque_msg(torque_desired)))
      self._set_prev_torque(torque_desired + delta)
      self.safety.set_volkswagen_torque_driver(-driver_torque, -driver_torque)
      self.assertFalse(self.safety.safety_tx_hook(self._torque_msg(torque_desired + delta)))

      self._set_prev_torque(MAX_STEER * sign)
      self.safety.set_volkswagen_torque_driver(-MAX_STEER * sign, -MAX_STEER * sign)
      self.assertTrue(self.safety.safety_tx_hook(self._torque_msg((MAX_STEER - MAX_RATE_DOWN) * sign)))
      self._set_prev_torque(MAX_STEER * sign)
      self.safety.set_volkswagen_torque_driver(-MAX_STEER * sign, -MAX_STEER * sign)
      self.assertTrue(self.safety.safety_tx_hook(self._torque_msg(0)))
      self._set_prev_torque(MAX_STEER * sign)
      self.safety.set_volkswagen_torque_driver(-MAX_STEER * sign, -MAX_STEER * sign)
      self.assertFalse(self.safety.safety_tx_hook(self._torque_msg((MAX_STEER - MAX_RATE_DOWN + 1) * sign)))


  def test_realtime_limits(self):
    self.safety.set_controls_allowed(True)

    for sign in [-1, 1]:
      self.safety.init_tests_volkswagen()
      self._set_prev_torque(0)
      self.safety.set_volkswagen_torque_driver(0, 0)
      for t in np.arange(0, MAX_RT_DELTA, 1):
        t *= sign
        self.assertTrue(self.safety.safety_tx_hook(self._torque_msg(t)))
      self.assertFalse(self.safety.safety_tx_hook(self._torque_msg(sign * (MAX_RT_DELTA + 1))))

      self._set_prev_torque(0)
      for t in np.arange(0, MAX_RT_DELTA, 1):
        t *= sign
        self.assertTrue(self.safety.safety_tx_hook(self._torque_msg(t)))

      # Increase timer to update rt_torque_last
      self.safety.set_timer(RT_INTERVAL + 1)
      self.assertTrue(self.safety.safety_tx_hook(self._torque_msg(sign * (MAX_RT_DELTA - 1))))
      self.assertTrue(self.safety.safety_tx_hook(self._torque_msg(sign * (MAX_RT_DELTA + 1))))


  def test_fwd_hook(self):
    buss = list(range(0x0, 0x3))
    msgs = list(range(0x1, 0x800))
    blocked_msgs_0to2 = []
    blocked_msgs_2to0 = [0x126, 0x397]
    for b in buss:
      for m in msgs:
        if b == 0:
          fwd_bus = -1 if m in blocked_msgs_0to2 else 2
        elif b == 1:
          fwd_bus = -1
        elif b == 2:
          fwd_bus = -1 if m in blocked_msgs_2to0 else 0

        # assume len 8
        self.assertEqual(fwd_bus, self.safety.safety_fwd_hook(b, make_msg(b, m, 8)))


if __name__ == "__main__":
  unittest.main()
