#!/usr/bin/env python3
import unittest
import numpy as np
import crcmod
from panda import Panda
from panda.tests.safety import libpandasafety_py
from panda.tests.safety.common import StdTest, make_msg, MAX_WRONG_COUNTERS

MAX_RATE_UP = 4
MAX_RATE_DOWN = 10
MAX_STEER = 300
MAX_RT_DELTA = 75
RT_INTERVAL = 250000

DRIVER_TORQUE_ALLOWANCE = 80
DRIVER_TORQUE_FACTOR = 3

MSG_ESP_19 = 0xB2       # RX from ABS, for wheel speeds
MSG_EPS_01 = 0x9F       # RX from EPS, for driver steering torque
MSG_ESP_05 = 0x106      # RX from ABS, for brake light state
MSG_TSK_06 = 0x120      # RX from ECU, for ACC status from drivetrain coordinator
MSG_MOTOR_20 = 0x121    # RX from ECU, for driver throttle input
MSG_HCA_01 = 0x126      # TX by OP, Heading Control Assist steering torque
MSG_GRA_ACC_01 = 0x12B  # TX by OP, ACC control buttons for cancel/resume
MSG_LDW_02 = 0x397      # TX by OP, Lane line recognition and text alerts

# Transmit of GRA_ACC_01 is allowed on bus 0 and 2 to keep compatibility with gateway and camera integration
TX_MSGS = [[MSG_HCA_01, 0], [MSG_GRA_ACC_01, 0], [MSG_GRA_ACC_01, 2], [MSG_LDW_02, 0]]

def sign(a):
  if a > 0:
    return 1
  else:
    return -1

# Python crcmod works differently somehow from every other CRC calculator. The
# implied leading 1 on the polynomial isn't a problem, but to get the right
# result for CRC-8H2F/AUTOSAR, we have to feed it initCrc 0x00 instead of 0xFF.
volkswagen_crc_8h2f = crcmod.mkCrcFun(0x12F, initCrc=0x00, rev=False, xorOut=0xFF)

def volkswagen_mqb_crc(msg, addr, len_msg):
  # This is CRC-8H2F/AUTOSAR with a twist. See the OpenDBC implementation of
  # this algorithm for a version with explanatory comments.
  msg_bytes = msg.RDLR.to_bytes(4, 'little') + msg.RDHR.to_bytes(4, 'little')
  counter = (msg.RDLR & 0xF00) >> 8
  if addr == MSG_EPS_01:
    magic_pad = b'\xF5\xF5\xF5\xF5\xF5\xF5\xF5\xF5\xF5\xF5\xF5\xF5\xF5\xF5\xF5\xF5'[counter]
  elif addr == MSG_ESP_05:
    magic_pad = b'\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07'[counter]
  elif addr == MSG_TSK_06:
    magic_pad = b'\xC4\xE2\x4F\xE4\xF8\x2F\x56\x81\x9F\xE5\x83\x44\x05\x3F\x97\xDF'[counter]
  elif addr == MSG_MOTOR_20:
    magic_pad = b'\xE9\x65\xAE\x6B\x7B\x35\xE5\x5F\x4E\xC7\x86\xA2\xBB\xDD\xEB\xB4'[counter]
  elif addr == MSG_HCA_01:
    magic_pad = b'\xDA\xDA\xDA\xDA\xDA\xDA\xDA\xDA\xDA\xDA\xDA\xDA\xDA\xDA\xDA\xDA'[counter]
  elif addr == MSG_GRA_ACC_01:
    magic_pad = b'\x6A\x38\xB4\x27\x22\xEF\xE1\xBB\xF8\x80\x84\x49\xC7\x9E\x1E\x2B'[counter]
  else:
    magic_pad = None
  return volkswagen_crc_8h2f(msg_bytes[1:len_msg] + magic_pad.to_bytes(1, 'little'))

class TestVolkswagenMqbSafety(unittest.TestCase):
  cnt_eps_01 = 0
  cnt_esp_05 = 0
  cnt_tsk_06 = 0
  cnt_motor_20 = 0
  cnt_hca_01 = 0
  cnt_gra_acc_01 = 0

  @classmethod
  def setUp(cls):
    cls.safety = libpandasafety_py.libpandasafety
    cls.safety.set_safety_hooks(Panda.SAFETY_VOLKSWAGEN_MQB, 0)
    cls.safety.init_tests_volkswagen()

  def _set_prev_torque(self, t):
    self.safety.set_volkswagen_desired_torque_last(t)
    self.safety.set_volkswagen_rt_torque_last(t)

  # Wheel speeds _esp_19_msg
  def _speed_msg(self, speed):
    wheel_speed_scaled = int(speed / 0.0075)
    to_send = make_msg(0, MSG_ESP_19)
    to_send[0].RDLR = wheel_speed_scaled | (wheel_speed_scaled << 16)
    to_send[0].RDHR = wheel_speed_scaled | (wheel_speed_scaled << 16)
    return to_send

  # Brake light switch _esp_05_msg
  def _brake_msg(self, brake):
    to_send = make_msg(0, MSG_ESP_05)
    to_send[0].RDLR = (0x1 << 26) if brake else 0
    to_send[0].RDLR |= (self.cnt_esp_05 % 16) << 8
    to_send[0].RDLR |= volkswagen_mqb_crc(to_send[0], MSG_ESP_05, 8)
    self.__class__.cnt_esp_05 += 1
    return to_send

  # Driver steering input torque
  def _eps_01_msg(self, torque):
    to_send = make_msg(0, MSG_EPS_01)
    t = abs(torque)
    to_send[0].RDHR = ((t & 0x1FFF) << 8)
    if torque < 0:
      to_send[0].RDHR |= 0x1 << 23
    to_send[0].RDLR |= (self.cnt_eps_01 % 16) << 8
    to_send[0].RDLR |= volkswagen_mqb_crc(to_send[0], MSG_EPS_01, 8)
    self.__class__.cnt_eps_01 += 1
    return to_send

  # openpilot steering output torque
  def _hca_01_msg(self, torque):
    to_send = make_msg(0, MSG_HCA_01)
    t = abs(torque)
    to_send[0].RDLR = (t & 0xFFF) << 16
    if torque < 0:
      to_send[0].RDLR |= 0x1 << 31
    to_send[0].RDLR |= (self.cnt_hca_01 % 16) << 8
    to_send[0].RDLR |= volkswagen_mqb_crc(to_send[0], MSG_HCA_01, 8)
    self.__class__.cnt_hca_01 += 1
    return to_send

  # ACC engagement status
  def _tsk_06_msg(self, status):
    to_send = make_msg(0, MSG_TSK_06)
    to_send[0].RDLR = (status & 0x7) << 24
    to_send[0].RDLR |= (self.cnt_tsk_06 % 16) << 8
    to_send[0].RDLR |= volkswagen_mqb_crc(to_send[0], MSG_TSK_06, 8)
    self.__class__.cnt_tsk_06 += 1
    return to_send

  # Driver throttle input
  def _motor_20_msg(self, gas):
    to_send = make_msg(0, MSG_MOTOR_20)
    to_send[0].RDLR = (gas & 0xFF) << 12
    to_send[0].RDLR |= (self.cnt_motor_20 % 16) << 8
    to_send[0].RDLR |= volkswagen_mqb_crc(to_send[0], MSG_MOTOR_20, 8)
    self.__class__.cnt_motor_20 += 1
    return to_send

  # Cruise control buttons
  def _gra_acc_01_msg(self, bit):
    to_send = make_msg(2, MSG_GRA_ACC_01)
    to_send[0].RDLR = 1 << bit
    to_send[0].RDLR |= (self.cnt_gra_acc_01 % 16) << 8
    to_send[0].RDLR |= volkswagen_mqb_crc(to_send[0], MSG_GRA_ACC_01, 8)
    self.__class__.cnt_gra_acc_01 += 1
    return to_send

  def test_spam_can_buses(self):
    StdTest.test_spam_can_buses(self, TX_MSGS)

  def test_relay_malfunction(self):
    StdTest.test_relay_malfunction(self, MSG_HCA_01)

  def test_prev_gas(self):
    for g in range(0, 256):
      self.safety.safety_rx_hook(self._motor_20_msg(g))
      self.assertEqual(True if g > 0 else False, self.safety.get_gas_pressed_prev())

  def test_default_controls_not_allowed(self):
    self.assertFalse(self.safety.get_controls_allowed())

  def test_enable_control_allowed_from_cruise(self):
    self.safety.set_controls_allowed(0)
    self.safety.safety_rx_hook(self._tsk_06_msg(3))
    self.assertTrue(self.safety.get_controls_allowed())

  def test_disable_control_allowed_from_cruise(self):
    self.safety.set_controls_allowed(1)
    self.safety.safety_rx_hook(self._tsk_06_msg(1))
    self.assertFalse(self.safety.get_controls_allowed())

  def test_sample_speed(self):
    # Stationary
    self.safety.safety_rx_hook(self._speed_msg(0))
    self.assertEqual(0, self.safety.get_volkswagen_moving())
    # 1 km/h, just under 0.3 m/s safety grace threshold
    self.safety.safety_rx_hook(self._speed_msg(1))
    self.assertEqual(0, self.safety.get_volkswagen_moving())
    # 2 km/h, just over 0.3 m/s safety grace threshold
    self.safety.safety_rx_hook(self._speed_msg(2))
    self.assertEqual(1, self.safety.get_volkswagen_moving())
    # 144 km/h, openpilot V_CRUISE_MAX
    self.safety.safety_rx_hook(self._speed_msg(144))
    self.assertEqual(1, self.safety.get_volkswagen_moving())

  def test_prev_brake(self):
    self.assertFalse(self.safety.get_brake_pressed_prev())
    self.safety.safety_rx_hook(self._brake_msg(True))
    self.assertTrue(self.safety.get_brake_pressed_prev())

  def test_brake_disengage(self):
    StdTest.test_allow_brake_at_zero_speed(self)
    StdTest.test_not_allow_brake_when_moving(self, 1)

  def test_disengage_on_gas(self):
    self.safety.safety_rx_hook(self._motor_20_msg(0))
    self.safety.set_controls_allowed(True)
    self.safety.safety_rx_hook(self._motor_20_msg(1))
    self.assertFalse(self.safety.get_controls_allowed())

  def test_allow_engage_with_gas_pressed(self):
    self.safety.safety_rx_hook(self._motor_20_msg(1))
    self.safety.set_controls_allowed(True)
    self.safety.safety_rx_hook(self._motor_20_msg(1))
    self.assertTrue(self.safety.get_controls_allowed())
    self.safety.safety_rx_hook(self._motor_20_msg(1))
    self.assertTrue(self.safety.get_controls_allowed())

  def test_steer_safety_check(self):
    for enabled in [0, 1]:
      for t in range(-500, 500):
        self.safety.set_controls_allowed(enabled)
        self._set_prev_torque(t)
        if abs(t) > MAX_STEER or (not enabled and abs(t) > 0):
          self.assertFalse(self.safety.safety_tx_hook(self._hca_01_msg(t)))
        else:
          self.assertTrue(self.safety.safety_tx_hook(self._hca_01_msg(t)))

  def test_manually_enable_controls_allowed(self):
    StdTest.test_manually_enable_controls_allowed(self)

  def test_spam_cancel_safety_check(self):
    BIT_CANCEL = 13
    BIT_RESUME = 19
    BIT_SET = 16
    self.safety.set_controls_allowed(0)
    self.assertTrue(self.safety.safety_tx_hook(self._gra_acc_01_msg(BIT_CANCEL)))
    self.assertFalse(self.safety.safety_tx_hook(self._gra_acc_01_msg(BIT_RESUME)))
    self.assertFalse(self.safety.safety_tx_hook(self._gra_acc_01_msg(BIT_SET)))
    # do not block resume if we are engaged already
    self.safety.set_controls_allowed(1)
    self.assertTrue(self.safety.safety_tx_hook(self._gra_acc_01_msg(BIT_RESUME)))

  def test_non_realtime_limit_up(self):
    self.safety.set_volkswagen_torque_driver(0, 0)
    self.safety.set_controls_allowed(True)

    self._set_prev_torque(0)
    self.assertTrue(self.safety.safety_tx_hook(self._hca_01_msg(MAX_RATE_UP)))
    self._set_prev_torque(0)
    self.assertTrue(self.safety.safety_tx_hook(self._hca_01_msg(-MAX_RATE_UP)))

    self._set_prev_torque(0)
    self.assertFalse(self.safety.safety_tx_hook(self._hca_01_msg(MAX_RATE_UP + 1)))
    self.safety.set_controls_allowed(True)
    self._set_prev_torque(0)
    self.assertFalse(self.safety.safety_tx_hook(self._hca_01_msg(-MAX_RATE_UP - 1)))

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
        self.assertTrue(self.safety.safety_tx_hook(self._hca_01_msg(MAX_STEER * sign)))

      self.safety.set_volkswagen_torque_driver(DRIVER_TORQUE_ALLOWANCE + 1, DRIVER_TORQUE_ALLOWANCE + 1)
      self.assertFalse(self.safety.safety_tx_hook(self._hca_01_msg(-MAX_STEER)))

    # spot check some individual cases
    for sign in [-1, 1]:
      driver_torque = (DRIVER_TORQUE_ALLOWANCE + 10) * sign
      torque_desired = (MAX_STEER - 10 * DRIVER_TORQUE_FACTOR) * sign
      delta = 1 * sign
      self._set_prev_torque(torque_desired)
      self.safety.set_volkswagen_torque_driver(-driver_torque, -driver_torque)
      self.assertTrue(self.safety.safety_tx_hook(self._hca_01_msg(torque_desired)))
      self._set_prev_torque(torque_desired + delta)
      self.safety.set_volkswagen_torque_driver(-driver_torque, -driver_torque)
      self.assertFalse(self.safety.safety_tx_hook(self._hca_01_msg(torque_desired + delta)))

      self._set_prev_torque(MAX_STEER * sign)
      self.safety.set_volkswagen_torque_driver(-MAX_STEER * sign, -MAX_STEER * sign)
      self.assertTrue(self.safety.safety_tx_hook(self._hca_01_msg((MAX_STEER - MAX_RATE_DOWN) * sign)))
      self._set_prev_torque(MAX_STEER * sign)
      self.safety.set_volkswagen_torque_driver(-MAX_STEER * sign, -MAX_STEER * sign)
      self.assertTrue(self.safety.safety_tx_hook(self._hca_01_msg(0)))
      self._set_prev_torque(MAX_STEER * sign)
      self.safety.set_volkswagen_torque_driver(-MAX_STEER * sign, -MAX_STEER * sign)
      self.assertFalse(self.safety.safety_tx_hook(self._hca_01_msg((MAX_STEER - MAX_RATE_DOWN + 1) * sign)))

  def test_realtime_limits(self):
    self.safety.set_controls_allowed(True)

    for sign in [-1, 1]:
      self.safety.init_tests_volkswagen()
      self._set_prev_torque(0)
      self.safety.set_volkswagen_torque_driver(0, 0)
      for t in np.arange(0, MAX_RT_DELTA, 1):
        t *= sign
        self.assertTrue(self.safety.safety_tx_hook(self._hca_01_msg(t)))
      self.assertFalse(self.safety.safety_tx_hook(self._hca_01_msg(sign * (MAX_RT_DELTA + 1))))

      self._set_prev_torque(0)
      for t in np.arange(0, MAX_RT_DELTA, 1):
        t *= sign
        self.assertTrue(self.safety.safety_tx_hook(self._hca_01_msg(t)))

      # Increase timer to update rt_torque_last
      self.safety.set_timer(RT_INTERVAL + 1)
      self.assertTrue(self.safety.safety_tx_hook(self._hca_01_msg(sign * (MAX_RT_DELTA - 1))))
      self.assertTrue(self.safety.safety_tx_hook(self._hca_01_msg(sign * (MAX_RT_DELTA + 1))))

  def test_torque_measurements(self):
    self.safety.safety_rx_hook(self._eps_01_msg(50))
    self.safety.safety_rx_hook(self._eps_01_msg(-50))
    self.safety.safety_rx_hook(self._eps_01_msg(0))
    self.safety.safety_rx_hook(self._eps_01_msg(0))
    self.safety.safety_rx_hook(self._eps_01_msg(0))
    self.safety.safety_rx_hook(self._eps_01_msg(0))

    self.assertEqual(-50, self.safety.get_volkswagen_torque_driver_min())
    self.assertEqual(50, self.safety.get_volkswagen_torque_driver_max())

    self.safety.safety_rx_hook(self._eps_01_msg(0))
    self.assertEqual(0, self.safety.get_volkswagen_torque_driver_max())
    self.assertEqual(-50, self.safety.get_volkswagen_torque_driver_min())

    self.safety.safety_rx_hook(self._eps_01_msg(0))
    self.assertEqual(0, self.safety.get_volkswagen_torque_driver_max())
    self.assertEqual(0, self.safety.get_volkswagen_torque_driver_min())

  def test_rx_hook(self):
    # checksum checks
    # TODO: Would be ideal to check ESP_19 as well, but it has no checksum
    # or counter, and I'm not sure if we can easily validate Panda's simple
    # temporal reception-rate check here.
    for msg in [MSG_EPS_01, MSG_ESP_05, MSG_TSK_06, MSG_MOTOR_20]:
      self.safety.set_controls_allowed(1)
      if msg == MSG_EPS_01:
        to_push = self._eps_01_msg(0)
      if msg == MSG_ESP_05:
        to_push = self._brake_msg(False)
      if msg == MSG_TSK_06:
        to_push = self._tsk_06_msg(3)
      if msg == MSG_MOTOR_20:
        to_push = self._motor_20_msg(0)
      self.assertTrue(self.safety.safety_rx_hook(to_push))
      to_push[0].RDHR ^= 0xFF
      self.assertFalse(self.safety.safety_rx_hook(to_push))
      self.assertFalse(self.safety.get_controls_allowed())

    # counter
    # reset wrong_counters to zero by sending valid messages
    for i in range(MAX_WRONG_COUNTERS + 1):
      self.__class__.cnt_eps_01 += 1
      self.__class__.cnt_esp_05 += 1
      self.__class__.cnt_tsk_06 += 1
      self.__class__.cnt_motor_20 += 1
      if i < MAX_WRONG_COUNTERS:
        self.safety.set_controls_allowed(1)
        self.safety.safety_rx_hook(self._eps_01_msg(0))
        self.safety.safety_rx_hook(self._brake_msg(False))
        self.safety.safety_rx_hook(self._tsk_06_msg(3))
        self.safety.safety_rx_hook(self._motor_20_msg(0))
      else:
        self.assertFalse(self.safety.safety_rx_hook(self._eps_01_msg(0)))
        self.assertFalse(self.safety.safety_rx_hook(self._brake_msg(False)))
        self.assertFalse(self.safety.safety_rx_hook(self._tsk_06_msg(3)))
        self.assertFalse(self.safety.safety_rx_hook(self._motor_20_msg(0)))
        self.assertFalse(self.safety.get_controls_allowed())

    # restore counters for future tests with a couple of good messages
    for i in range(2):
      self.safety.set_controls_allowed(1)
      self.safety.safety_rx_hook(self._eps_01_msg(0))
      self.safety.safety_rx_hook(self._brake_msg(False))
      self.safety.safety_rx_hook(self._tsk_06_msg(3))
      self.safety.safety_rx_hook(self._motor_20_msg(0))
    self.assertTrue(self.safety.get_controls_allowed())

  def test_fwd_hook(self):
    buss = list(range(0x0, 0x3))
    msgs = list(range(0x1, 0x800))
    blocked_msgs_0to2 = []
    blocked_msgs_2to0 = [MSG_HCA_01, MSG_LDW_02]
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
