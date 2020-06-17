#!/usr/bin/env python3
import unittest
import numpy as np
from panda import Panda
from panda.tests.safety import libpandasafety_py
import panda.tests.safety.common as common
from panda.tests.safety.common import make_msg, MAX_WRONG_COUNTERS

MAX_RATE_UP = 4
MAX_RATE_DOWN = 10
MAX_STEER = 300
MAX_RT_DELTA = 75
RT_INTERVAL = 250000

DRIVER_TORQUE_ALLOWANCE = 80
DRIVER_TORQUE_FACTOR = 3

MSG_LENKHILFE_3 = 0x0D0  # RX from EPS, for steering angle and driver steering torque
MSG_HCA_1 = 0x0D2        # TX by OP, Heading Control Assist steering torque
MSG_MOTOR_2 = 0x288      # RX from ECU, for CC state and brake switch state
MSG_MOTOR_3 = 0x380      # RX from ECU, for driver throttle input
MSG_GRA_NEU = 0x38A      # TX by OP, ACC control buttons for cancel/resume
MSG_BREMSE_3 = 0x4A0     # RX from ABS, for wheel speeds
MSG_LDW_1 = 0x5BE        # TX by OP, Lane line recognition and text alerts


def volkswagen_pq_checksum(msg, addr, len_msg):
  msg_bytes = msg.RDLR.to_bytes(4, 'little') + msg.RDHR.to_bytes(4, 'little')
  msg_bytes = msg_bytes[1:len_msg]

  checksum = 0
  for i in msg_bytes:
    checksum ^= i
  return checksum

class TestVolkswagenPqSafety(common.PandaSafetyTest):
  cruise_engaged = False
  brake_pressed = False
  cnt_lenkhilfe_3 = 0
  cnt_hca_1 = 0

  # Transmit of GRA_Neu is allowed on bus 0 and 2 to keep compatibility with gateway and camera integration
  TX_MSGS = [[MSG_HCA_1, 0], [MSG_GRA_NEU, 0], [MSG_GRA_NEU, 2], [MSG_LDW_1, 0]]
  STANDSTILL_THRESHOLD = 1
  RELAY_MALFUNCTION_ADDR = MSG_HCA_1
  RELAY_MALFUNCTION_BUS = 0
  FWD_BLACKLISTED_ADDRS = {2: [MSG_HCA_1, MSG_LDW_1]}
  FWD_BUS_LOOKUP = {0: 2, 2: 0}

  def setUp(self):
    self.safety = libpandasafety_py.libpandasafety
    self.safety.set_safety_hooks(Panda.SAFETY_VOLKSWAGEN_PQ, 0)
    self.safety.init_tests()

  # override these inherited tests from PandaSafetyTest
  def test_cruise_engaged_prev(self): pass

  def _set_prev_torque(self, t):
    self.safety.set_desired_torque_last(t)
    self.safety.set_rt_torque_last(t)

  # Wheel speeds (Bremse_3)
  def _speed_msg(self, speed):
    wheel_speed_scaled = int(speed / 0.01)
    to_send = make_msg(0, MSG_BREMSE_3)
    to_send[0].RDLR = (wheel_speed_scaled | (wheel_speed_scaled << 16)) << 1
    to_send[0].RDHR = (wheel_speed_scaled | (wheel_speed_scaled << 16)) << 1
    return to_send

  # Brake light switch (shared message Motor_2)
  def _brake_msg(self, brake):
    to_send = make_msg(0, MSG_MOTOR_2)
    to_send[0].RDLR = (0x1 << 16) if brake else 0
    # since this siganl's used for engagement status, preserve current state
    to_send[0].RDLR |= (self.safety.get_controls_allowed() & 0x3) << 22
    return to_send

  # ACC engaged status (shared message Motor_2)
  def _pcm_status_msg(self, cruise):
    self.__class__.cruise_engaged = cruise
    return self._motor_2_msg()

  # Driver steering input torque
  def _lenkhilfe_3_msg(self, torque):
    to_send = make_msg(0, MSG_LENKHILFE_3, 6)
    t = abs(torque)
    to_send[0].RDLR = ((t & 0x3FF) << 16)
    if torque < 0:
      to_send[0].RDLR |= 0x1 << 26
    to_send[0].RDLR |= (self.cnt_lenkhilfe_3 % 16) << 12
    to_send[0].RDLR |= volkswagen_pq_checksum(to_send[0], MSG_LENKHILFE_3, 8)
    self.__class__.cnt_lenkhilfe_3 += 1
    return to_send

  # openpilot steering output torque
  def _hca_1_msg(self, torque):
    to_send = make_msg(0, MSG_HCA_1, 5)
    t = abs(torque) << 5  # DBC scale from centi-Nm to PQ network (approximated)
    to_send[0].RDLR = (t & 0x7FFF) << 16
    if torque < 0:
      to_send[0].RDLR |= 0x1 << 31
    to_send[0].RDLR |= (self.cnt_hca_1 % 16) << 8
    to_send[0].RDLR |= volkswagen_pq_checksum(to_send[0], MSG_HCA_1, 8)
    self.__class__.cnt_hca_1 += 1
    return to_send

  # ACC engagement and brake light switch status
  # Called indirectly for compatibility with common.py tests
  def _motor_2_msg(self):
    to_send = make_msg(0, MSG_MOTOR_2)
    to_send[0].RDLR = (0x1 << 16) if self.__class__.brake_pressed else 0
    to_send[0].RDLR |= (self.__class__.cruise_engaged & 0x3) << 22
    return to_send

  # Driver throttle input (motor_3)
  def _gas_msg(self, gas):
    to_send = make_msg(0, MSG_MOTOR_3)
    to_send[0].RDLR = (gas & 0xFF) << 16
    return to_send

  # Cruise control buttons
  def _gra_neu_msg(self, bit):
    to_send = make_msg(2, MSG_GRA_NEU, 4)
    to_send[0].RDLR = 1 << bit
    to_send[0].RDLR |= volkswagen_pq_checksum(to_send[0], MSG_GRA_NEU, 8)
    return to_send

  def test_steer_safety_check(self):
    for enabled in [0, 1]:
      for t in range(-500, 500):
        self.safety.set_controls_allowed(enabled)
        self._set_prev_torque(t)
        if abs(t) > MAX_STEER or (not enabled and abs(t) > 0):
          self.assertFalse(self._tx(self._hca_1_msg(t)))
        else:
          self.assertTrue(self._tx(self._hca_1_msg(t)))

  def test_spam_cancel_safety_check(self):
    BIT_CANCEL = 9
    BIT_SET = 16
    BIT_RESUME = 17
    self.safety.set_controls_allowed(0)
    self.assertTrue(self._tx(self._gra_neu_msg(BIT_CANCEL)))
    self.assertFalse(self._tx(self._gra_neu_msg(BIT_RESUME)))
    self.assertFalse(self._tx(self._gra_neu_msg(BIT_SET)))
    # do not block resume if we are engaged already
    self.safety.set_controls_allowed(1)
    self.assertTrue(self._tx(self._gra_neu_msg(BIT_RESUME)))

  def test_non_realtime_limit_up(self):
    self.safety.set_torque_driver(0, 0)
    self.safety.set_controls_allowed(True)

    self._set_prev_torque(0)
    self.assertTrue(self._tx(self._hca_1_msg(MAX_RATE_UP)))
    self._set_prev_torque(0)
    self.assertTrue(self._tx(self._hca_1_msg(-MAX_RATE_UP)))

    self._set_prev_torque(0)
    self.assertFalse(self._tx(self._hca_1_msg(MAX_RATE_UP + 1)))
    self.safety.set_controls_allowed(True)
    self._set_prev_torque(0)
    self.assertFalse(self._tx(self._hca_1_msg(-MAX_RATE_UP - 1)))

  def test_non_realtime_limit_down(self):
    self.safety.set_torque_driver(0, 0)
    self.safety.set_controls_allowed(True)

  def test_against_torque_driver(self):
    self.safety.set_controls_allowed(True)

    for sign in [-1, 1]:
      for t in np.arange(0, DRIVER_TORQUE_ALLOWANCE + 1, 1):
        t *= -sign
        self.safety.set_torque_driver(t, t)
        self._set_prev_torque(MAX_STEER * sign)
        self.assertTrue(self._tx(self._hca_1_msg(MAX_STEER * sign)))

      self.safety.set_torque_driver(DRIVER_TORQUE_ALLOWANCE + 1, DRIVER_TORQUE_ALLOWANCE + 1)
      self.assertFalse(self._tx(self._hca_1_msg(-MAX_STEER)))

    # spot check some individual cases
    for sign in [-1, 1]:
      driver_torque = (DRIVER_TORQUE_ALLOWANCE + 10) * sign
      torque_desired = (MAX_STEER - 10 * DRIVER_TORQUE_FACTOR) * sign
      delta = 1 * sign
      self._set_prev_torque(torque_desired)
      self.safety.set_torque_driver(-driver_torque, -driver_torque)
      self.assertTrue(self._tx(self._hca_1_msg(torque_desired)))
      self._set_prev_torque(torque_desired + delta)
      self.safety.set_torque_driver(-driver_torque, -driver_torque)
      self.assertFalse(self._tx(self._hca_1_msg(torque_desired + delta)))

      self._set_prev_torque(MAX_STEER * sign)
      self.safety.set_torque_driver(-MAX_STEER * sign, -MAX_STEER * sign)
      self.assertTrue(self._tx(self._hca_1_msg((MAX_STEER - MAX_RATE_DOWN) * sign)))
      self._set_prev_torque(MAX_STEER * sign)
      self.safety.set_torque_driver(-MAX_STEER * sign, -MAX_STEER * sign)
      self.assertTrue(self._tx(self._hca_1_msg(0)))
      self._set_prev_torque(MAX_STEER * sign)
      self.safety.set_torque_driver(-MAX_STEER * sign, -MAX_STEER * sign)
      self.assertFalse(self._tx(self._hca_1_msg((MAX_STEER - MAX_RATE_DOWN + 1) * sign)))

  def test_realtime_limits(self):
    self.safety.set_controls_allowed(True)

    for sign in [-1, 1]:
      self.safety.init_tests()
      self._set_prev_torque(0)
      self.safety.set_torque_driver(0, 0)
      for t in np.arange(0, MAX_RT_DELTA, 1):
        t *= sign
        self.assertTrue(self._tx(self._hca_1_msg(t)))
      self.assertFalse(self._tx(self._hca_1_msg(sign * (MAX_RT_DELTA + 1))))

      self._set_prev_torque(0)
      for t in np.arange(0, MAX_RT_DELTA, 1):
        t *= sign
        self.assertTrue(self._tx(self._hca_1_msg(t)))

      # Increase timer to update rt_torque_last
      self.safety.set_timer(RT_INTERVAL + 1)
      self.assertTrue(self._tx(self._hca_1_msg(sign * (MAX_RT_DELTA - 1))))
      self.assertTrue(self._tx(self._hca_1_msg(sign * (MAX_RT_DELTA + 1))))

  def test_torque_measurements(self):
    self._rx(self._lenkhilfe_3_msg(50))
    self._rx(self._lenkhilfe_3_msg(-50))
    self._rx(self._lenkhilfe_3_msg(0))
    self._rx(self._lenkhilfe_3_msg(0))
    self._rx(self._lenkhilfe_3_msg(0))
    self._rx(self._lenkhilfe_3_msg(0))

    self.assertEqual(-50, self.safety.get_torque_driver_min())
    self.assertEqual(50, self.safety.get_torque_driver_max())

    self._rx(self._lenkhilfe_3_msg(0))
    self.assertEqual(0, self.safety.get_torque_driver_max())
    self.assertEqual(-50, self.safety.get_torque_driver_min())

    self._rx(self._lenkhilfe_3_msg(0))
    self.assertEqual(0, self.safety.get_torque_driver_max())
    self.assertEqual(0, self.safety.get_torque_driver_min())

  def test_rx_hook(self):
    # checksum checks
    # TODO: Would be ideal to check non-checksum non-counter messages as well,
    # but I'm not sure if we can easily validate Panda's simple temporal
    # reception-rate check here.
    for msg in [MSG_LENKHILFE_3]:
      self.safety.set_controls_allowed(1)
      if msg == MSG_LENKHILFE_3:
        to_push = self._lenkhilfe_3_msg(0)
      self.assertTrue(self._rx(to_push))
      to_push[0].RDHR ^= 0xFF
      self.assertFalse(self._rx(to_push))
      self.assertFalse(self.safety.get_controls_allowed())

    # counter
    # reset wrong_counters to zero by sending valid messages
    for i in range(MAX_WRONG_COUNTERS + 1):
      self.__class__.cnt_lenkhilfe_3 += 1
      if i < MAX_WRONG_COUNTERS:
        self.safety.set_controls_allowed(1)
        self._rx(self._lenkhilfe_3_msg(0))
      else:
        self.assertFalse(self._rx(self._lenkhilfe_3_msg(0)))
        self.assertFalse(self.safety.get_controls_allowed())

    # restore counters for future tests with a couple of good messages
    for i in range(2):
      self.safety.set_controls_allowed(1)
      self._rx(self._lenkhilfe_3_msg(0))
    self.assertTrue(self.safety.get_controls_allowed())


if __name__ == "__main__":
  unittest.main()
