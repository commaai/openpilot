#!/usr/bin/env python3
import unittest
import numpy as np
from panda import Panda
from panda.tests.safety import libpandasafety_py
import panda.tests.safety.common as common
from panda.tests.safety.common import make_msg

MAX_RATE_UP = 3
MAX_RATE_DOWN = 3
MAX_STEER = 261

MAX_RT_DELTA = 112
RT_INTERVAL = 250000

MAX_TORQUE_ERROR = 80


def chrysler_checksum(msg, len_msg):
  checksum = 0xFF
  for idx in range(0, len_msg-1):
    curr = (msg.RDLR >> (8*idx)) if idx < 4 else (msg.RDHR >> (8*(idx - 4)))
    curr &= 0xFF
    shift = 0x80
    for i in range(0, 8):
      bit_sum = curr & shift
      temp_chk = checksum & 0x80
      if (bit_sum != 0):
        bit_sum = 0x1C
        if (temp_chk != 0):
          bit_sum = 1
        checksum = checksum << 1
        temp_chk = checksum | 1
        bit_sum ^= temp_chk
      else:
        if (temp_chk != 0):
          bit_sum = 0x1D
        checksum = checksum << 1
        bit_sum ^= checksum
      checksum = bit_sum
      shift = shift >> 1
  return ~checksum & 0xFF

class TestChryslerSafety(common.PandaSafetyTest):
  cnt_torque_meas = 0
  cnt_gas = 0
  cnt_cruise = 0
  cnt_brake = 0

  TX_MSGS = [[571, 0], [658, 0], [678, 0]]
  STANDSTILL_THRESHOLD = 0
  RELAY_MALFUNCTION_ADDR = 0x292
  RELAY_MALFUNCTION_BUS = 0
  FWD_BLACKLISTED_ADDRS = {2: [658, 678]}
  FWD_BUS_LOOKUP = {0: 2, 2: 0}

  def setUp(self):
    self.safety = libpandasafety_py.libpandasafety
    self.safety.set_safety_hooks(Panda.SAFETY_CHRYSLER, 0)
    self.safety.init_tests_chrysler()

  def _button_msg(self, buttons):
    to_send = make_msg(0, 571)
    to_send[0].RDLR = buttons
    return to_send

  def _pcm_status_msg(self, active):
    to_send = make_msg(0, 500)
    to_send[0].RDLR = 0x380000 if active else 0
    to_send[0].RDHR |= (self.cnt_cruise % 16) << 20
    to_send[0].RDHR |= chrysler_checksum(to_send[0], 8) << 24
    self.__class__.cnt_cruise += 1
    return to_send

  def _speed_msg(self, speed):
    speed = int(speed / 0.071028)
    to_send = make_msg(0, 514, 4)
    to_send[0].RDLR = ((speed & 0xFF0) >> 4) + ((speed & 0xF) << 12) + \
                      ((speed & 0xFF0) << 12) + ((speed & 0xF) << 28)
    return to_send

  def _gas_msg(self, gas):
    to_send = make_msg(0, 308)
    to_send[0].RDHR = (gas & 0x7F) << 8
    to_send[0].RDHR |= (self.cnt_gas % 16) << 20
    self.__class__.cnt_gas += 1
    return to_send

  def _brake_msg(self, brake):
    to_send = make_msg(0, 320)
    to_send[0].RDLR = 5 if brake else 0
    to_send[0].RDHR |= (self.cnt_brake % 16) << 20
    to_send[0].RDHR |= chrysler_checksum(to_send[0], 8) << 24
    self.__class__.cnt_brake += 1
    return to_send

  def _set_prev_torque(self, t):
    self.safety.set_chrysler_desired_torque_last(t)
    self.safety.set_chrysler_rt_torque_last(t)
    self.safety.set_chrysler_torque_meas(t, t)

  def _torque_meas_msg(self, torque):
    to_send = make_msg(0, 544)
    to_send[0].RDHR = ((torque + 1024) >> 8) + (((torque + 1024) & 0xff) << 8)
    to_send[0].RDHR |= (self.cnt_torque_meas % 16) << 20
    to_send[0].RDHR |= chrysler_checksum(to_send[0], 8) << 24
    self.__class__.cnt_torque_meas += 1
    return to_send

  def _torque_msg(self, torque):
    to_send = make_msg(0, 0x292)
    to_send[0].RDLR = ((torque + 1024) >> 8) + (((torque + 1024) & 0xff) << 8)
    return to_send

  def test_steer_safety_check(self):
    for enabled in [0, 1]:
      for t in range(-MAX_STEER*2, MAX_STEER*2):
        self.safety.set_controls_allowed(enabled)
        self._set_prev_torque(t)
        if abs(t) > MAX_STEER or (not enabled and abs(t) > 0):
          self.assertFalse(self.safety.safety_tx_hook(self._torque_msg(t)))
        else:
          self.assertTrue(self.safety.safety_tx_hook(self._torque_msg(t)))
  
  # TODO: why does chrysler check if moving?
  def test_disengage_on_gas(self):
    self.safety.set_controls_allowed(1)
    self.safety.safety_rx_hook(self._speed_msg(2.2))
    self.safety.safety_rx_hook(self._gas_msg(1))
    self.assertTrue(self.safety.get_controls_allowed())
    self.safety.safety_rx_hook(self._gas_msg(0))
    self.safety.safety_rx_hook(self._speed_msg(2.3))
    self.safety.safety_rx_hook(self._gas_msg(1))
    self.assertFalse(self.safety.get_controls_allowed())

  def test_non_realtime_limit_up(self):
    self.safety.set_controls_allowed(True)

    self._set_prev_torque(0)
    self.assertTrue(self.safety.safety_tx_hook(self._torque_msg(MAX_RATE_UP)))

    self._set_prev_torque(0)
    self.assertFalse(self.safety.safety_tx_hook(self._torque_msg(MAX_RATE_UP + 1)))

  def test_non_realtime_limit_down(self):
    self.safety.set_controls_allowed(True)

    self.safety.set_chrysler_rt_torque_last(MAX_STEER)
    torque_meas = MAX_STEER - MAX_TORQUE_ERROR - 20
    self.safety.set_chrysler_torque_meas(torque_meas, torque_meas)
    self.safety.set_chrysler_desired_torque_last(MAX_STEER)
    self.assertTrue(self.safety.safety_tx_hook(self._torque_msg(MAX_STEER - MAX_RATE_DOWN)))

    self.safety.set_chrysler_rt_torque_last(MAX_STEER)
    self.safety.set_chrysler_torque_meas(torque_meas, torque_meas)
    self.safety.set_chrysler_desired_torque_last(MAX_STEER)
    self.assertFalse(self.safety.safety_tx_hook(self._torque_msg(MAX_STEER - MAX_RATE_DOWN + 1)))

  def test_exceed_torque_sensor(self):
    self.safety.set_controls_allowed(True)

    for sign in [-1, 1]:
      self._set_prev_torque(0)
      for t in np.arange(0, MAX_TORQUE_ERROR + 2, 2):  # step needs to be smaller than MAX_TORQUE_ERROR
        t *= sign
        self.assertTrue(self.safety.safety_tx_hook(self._torque_msg(t)))

      self.assertFalse(self.safety.safety_tx_hook(self._torque_msg(sign * (MAX_TORQUE_ERROR + 2))))

  def test_realtime_limit_up(self):
    self.safety.set_controls_allowed(True)

    for sign in [-1, 1]:
      self.safety.init_tests_chrysler()
      self._set_prev_torque(0)
      for t in np.arange(0, MAX_RT_DELTA+1, 1):
        t *= sign
        self.safety.set_chrysler_torque_meas(t, t)
        self.assertTrue(self.safety.safety_tx_hook(self._torque_msg(t)))
      self.assertFalse(self.safety.safety_tx_hook(self._torque_msg(sign * (MAX_RT_DELTA + 1))))

      self._set_prev_torque(0)
      for t in np.arange(0, MAX_RT_DELTA+1, 1):
        t *= sign
        self.safety.set_chrysler_torque_meas(t, t)
        self.assertTrue(self.safety.safety_tx_hook(self._torque_msg(t)))

      # Increase timer to update rt_torque_last
      self.safety.set_timer(RT_INTERVAL + 1)
      self.assertTrue(self.safety.safety_tx_hook(self._torque_msg(sign * MAX_RT_DELTA)))
      self.assertTrue(self.safety.safety_tx_hook(self._torque_msg(sign * (MAX_RT_DELTA + 1))))

  def test_torque_measurements(self):
    self.safety.safety_rx_hook(self._torque_meas_msg(50))
    self.safety.safety_rx_hook(self._torque_meas_msg(-50))
    self.safety.safety_rx_hook(self._torque_meas_msg(0))
    self.safety.safety_rx_hook(self._torque_meas_msg(0))
    self.safety.safety_rx_hook(self._torque_meas_msg(0))
    self.safety.safety_rx_hook(self._torque_meas_msg(0))

    self.assertEqual(-50, self.safety.get_chrysler_torque_meas_min())
    self.assertEqual(50, self.safety.get_chrysler_torque_meas_max())

    self.safety.safety_rx_hook(self._torque_meas_msg(0))
    self.assertEqual(0, self.safety.get_chrysler_torque_meas_max())
    self.assertEqual(-50, self.safety.get_chrysler_torque_meas_min())

    self.safety.safety_rx_hook(self._torque_meas_msg(0))
    self.assertEqual(0, self.safety.get_chrysler_torque_meas_max())
    self.assertEqual(0, self.safety.get_chrysler_torque_meas_min())

  def test_cancel_button(self):
    CANCEL = 1
    for b in range(0, 0x1ff):
      if b == CANCEL:
        self.assertTrue(self.safety.safety_tx_hook(self._button_msg(b)))
      else:
        self.assertFalse(self.safety.safety_tx_hook(self._button_msg(b)))


if __name__ == "__main__":
  unittest.main()
