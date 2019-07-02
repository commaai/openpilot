#!/usr/bin/env python2
import unittest
import numpy as np
import libpandasafety_py

MAX_RATE_UP = 3
MAX_RATE_DOWN = 7
MAX_STEER = 255

MAX_RT_DELTA = 112
RT_INTERVAL = 250000

DRIVER_TORQUE_ALLOWANCE = 50;
DRIVER_TORQUE_FACTOR = 2;

def twos_comp(val, bits):
  if val >= 0:
    return val
  else:
    return (2**bits) + val

def sign(a):
  if a > 0:
    return 1
  else:
    return -1

class TestHyundaiSafety(unittest.TestCase):
  @classmethod
  def setUp(cls):
    cls.safety = libpandasafety_py.libpandasafety
    cls.safety.safety_set_mode(7, 0)
    cls.safety.init_tests_hyundai()

  def _send_msg(self, bus, addr, length):
    to_send = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
    to_send[0].RIR = addr << 21
    to_send[0].RDTR = length
    to_send[0].RDTR = bus << 4
    return to_send

  def _button_msg(self, buttons):
    to_send = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
    to_send[0].RIR = 1265 << 21
    to_send[0].RDLR = buttons
    return to_send

  def _set_prev_torque(self, t):
    self.safety.set_hyundai_desired_torque_last(t)
    self.safety.set_hyundai_rt_torque_last(t)

  def _torque_driver_msg(self, torque):
    to_send = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
    to_send[0].RIR = 897 << 21
    to_send[0].RDLR = (torque + 2048) << 11
    return to_send

  def _torque_msg(self, torque):
    to_send = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
    to_send[0].RIR = 832 << 21
    to_send[0].RDLR = (torque + 1024) << 16
    return to_send

  def test_default_controls_not_allowed(self):
    self.assertFalse(self.safety.get_controls_allowed())

  def test_steer_safety_check(self):
    for enabled in [0, 1]:
      for t in range(-0x200, 0x200):
        self.safety.set_controls_allowed(enabled)
        self._set_prev_torque(t)
        if abs(t) > MAX_STEER or (not enabled and abs(t) > 0):
          self.assertFalse(self.safety.safety_tx_hook(self._torque_msg(t)))
        else:
          self.assertTrue(self.safety.safety_tx_hook(self._torque_msg(t)))

  def test_manually_enable_controls_allowed(self):
    self.safety.set_controls_allowed(1)
    self.assertTrue(self.safety.get_controls_allowed())
    self.safety.set_controls_allowed(0)
    self.assertFalse(self.safety.get_controls_allowed())

  def test_enable_control_allowed_from_cruise(self):
    to_push = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
    to_push[0].RIR = 1057 << 21
    to_push[0].RDLR = 1 << 13

    self.safety.safety_rx_hook(to_push)
    self.assertTrue(self.safety.get_controls_allowed())

  def test_disable_control_allowed_from_cruise(self):
    to_push = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
    to_push[0].RIR = 1057 << 21
    to_push[0].RDLR = 0

    self.safety.set_controls_allowed(1)
    self.safety.safety_rx_hook(to_push)
    self.assertFalse(self.safety.get_controls_allowed())

  def test_non_realtime_limit_up(self):
    self.safety.set_hyundai_torque_driver(0, 0)
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
    self.safety.set_hyundai_torque_driver(0, 0)
    self.safety.set_controls_allowed(True)

  def test_against_torque_driver(self):
    self.safety.set_controls_allowed(True)

    for sign in [-1, 1]:
      for t in np.arange(0, DRIVER_TORQUE_ALLOWANCE + 1, 1):
        t *= -sign
        self.safety.set_hyundai_torque_driver(t, t)
        self._set_prev_torque(MAX_STEER * sign)
        self.assertTrue(self.safety.safety_tx_hook(self._torque_msg(MAX_STEER * sign)))

      self.safety.set_hyundai_torque_driver(DRIVER_TORQUE_ALLOWANCE + 1, DRIVER_TORQUE_ALLOWANCE + 1)
      self.assertFalse(self.safety.safety_tx_hook(self._torque_msg(-MAX_STEER)))

    # spot check some individual cases
    for sign in [-1, 1]:
      driver_torque = (DRIVER_TORQUE_ALLOWANCE + 10) * sign
      torque_desired = (MAX_STEER - 10 * DRIVER_TORQUE_FACTOR) * sign
      delta = 1 * sign
      self._set_prev_torque(torque_desired)
      self.safety.set_hyundai_torque_driver(-driver_torque, -driver_torque)
      self.assertTrue(self.safety.safety_tx_hook(self._torque_msg(torque_desired)))
      self._set_prev_torque(torque_desired + delta)
      self.safety.set_hyundai_torque_driver(-driver_torque, -driver_torque)
      self.assertFalse(self.safety.safety_tx_hook(self._torque_msg(torque_desired + delta)))

      self._set_prev_torque(MAX_STEER * sign)
      self.safety.set_hyundai_torque_driver(-MAX_STEER * sign, -MAX_STEER * sign)
      self.assertTrue(self.safety.safety_tx_hook(self._torque_msg((MAX_STEER - MAX_RATE_DOWN) * sign)))
      self._set_prev_torque(MAX_STEER * sign)
      self.safety.set_hyundai_torque_driver(-MAX_STEER * sign, -MAX_STEER * sign)
      self.assertTrue(self.safety.safety_tx_hook(self._torque_msg(0)))
      self._set_prev_torque(MAX_STEER * sign)
      self.safety.set_hyundai_torque_driver(-MAX_STEER * sign, -MAX_STEER * sign)
      self.assertFalse(self.safety.safety_tx_hook(self._torque_msg((MAX_STEER - MAX_RATE_DOWN + 1) * sign)))


  def test_realtime_limits(self):
    self.safety.set_controls_allowed(True)

    for sign in [-1, 1]:
      self.safety.init_tests_hyundai()
      self._set_prev_torque(0)
      self.safety.set_hyundai_torque_driver(0, 0)
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


  #def test_spam_cancel_safety_check(self):
  #  RESUME_BTN = 1
  #  SET_BTN = 2
  #  CANCEL_BTN = 4
  #  BUTTON_MSG = 1265
  #  self.safety.set_controls_allowed(0)
  #  self.assertTrue(self.safety.safety_tx_hook(self._button_msg(CANCEL_BTN)))
  #  self.assertFalse(self.safety.safety_tx_hook(self._button_msg(RESUME_BTN)))
  #  self.assertFalse(self.safety.safety_tx_hook(self._button_msg(SET_BTN)))
  #  # do not block resume if we are engaged already
  #  self.safety.set_controls_allowed(1)
  #  self.assertTrue(self.safety.safety_tx_hook(self._button_msg(RESUME_BTN)))

  def test_fwd_hook(self):

    buss = range(0x0, 0x3)
    msgs = range(0x1, 0x800)
    hyundai_giraffe_switch_2 = [0, 1]

    self.safety.set_hyundai_camera_bus(2)
    for hgs in hyundai_giraffe_switch_2:
      self.safety.set_hyundai_giraffe_switch_2(hgs)
      blocked_msgs = [832]
      for b in buss:
        for m in msgs:
          if hgs:
            if b == 0:
              fwd_bus = 2
            elif b == 1:
              fwd_bus = -1
            elif b == 2:
              fwd_bus = -1 if m in blocked_msgs else 0
          else:
            fwd_bus = -1

          # assume len 8
          self.assertEqual(fwd_bus, self.safety.safety_fwd_hook(b, self._send_msg(b, m, 8)))


if __name__ == "__main__":
  unittest.main()
