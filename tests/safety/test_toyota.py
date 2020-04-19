#!/usr/bin/env python3
import unittest
import numpy as np
from panda import Panda
from panda.tests.safety import libpandasafety_py
import panda.tests.safety.common as common
from panda.tests.safety.common import CANPackerPanda, make_msg, UNSAFE_MODE

MAX_RATE_UP = 10
MAX_RATE_DOWN = 25
MAX_TORQUE = 1500

MAX_ACCEL = 1.5
MIN_ACCEL = -3.0

ISO_MAX_ACCEL = 2.0
ISO_MIN_ACCEL = -3.5

MAX_RT_DELTA = 375
RT_INTERVAL = 250000

MAX_TORQUE_ERROR = 350
INTERCEPTOR_THRESHOLD = 845

# Toyota gas gains are the same
def toyota_interceptor_msg(gas, addr):
  to_send = make_msg(0, addr, 6)
  to_send[0].RDLR = ((gas & 0xff) << 8) | ((gas & 0xff00) >> 8) | \
                    ((gas & 0xff) << 24) | ((gas & 0xff00) << 8)
  return to_send

class TestToyotaSafety(common.PandaSafetyTest):

  TX_MSGS = [[0x283, 0], [0x2E6, 0], [0x2E7, 0], [0x33E, 0], [0x344, 0], [0x365, 0], [0x366, 0], [0x4CB, 0],  # DSU bus 0
             [0x128, 1], [0x141, 1], [0x160, 1], [0x161, 1], [0x470, 1],  # DSU bus 1
             [0x2E4, 0], [0x411, 0], [0x412, 0], [0x343, 0], [0x1D2, 0],  # LKAS + ACC
             [0x200, 0], [0x750, 0]]  # interceptor + blindspot monitor
  STANDSTILL_THRESHOLD = 1  # 1kph
  RELAY_MALFUNCTION_ADDR = 0x2E4
  RELAY_MALFUNCTION_BUS = 0
  FWD_BLACKLISTED_ADDRS = {2: [0x2E4, 0x412, 0x191, 0x343]}
  FWD_BUS_LOOKUP = {0: 2, 2: 0}

  @classmethod
  def setUp(cls):
    cls.packer = CANPackerPanda("toyota_prius_2017_pt_generated")
    cls.safety = libpandasafety_py.libpandasafety
    cls.safety.set_safety_hooks(Panda.SAFETY_TOYOTA, 66)
    cls.safety.init_tests_toyota()

  def _set_prev_torque(self, t):
    self.safety.set_toyota_desired_torque_last(t)
    self.safety.set_toyota_rt_torque_last(t)
    self.safety.set_toyota_torque_meas(t, t)

  def _torque_meas_msg(self, torque):
    values = {"STEER_TORQUE_EPS": torque}
    return self.packer.make_can_msg_panda("STEER_TORQUE_SENSOR", 0, values)

  def _torque_msg(self, torque):
    values = {"STEER_TORQUE_CMD": torque}
    return self.packer.make_can_msg_panda("STEERING_LKA", 0, values)

  def _accel_msg(self, accel):
    values = {"ACCEL_CMD": accel}
    return self.packer.make_can_msg_panda("ACC_CONTROL", 0, values)

  def _speed_msg(self, s):
    values = {("WHEEL_SPEED_%s"%n): s for n in ["FR", "FL", "RR", "RL"]}
    return self.packer.make_can_msg_panda("WHEEL_SPEEDS", 0, values)

  def _brake_msg(self, pressed):
    values = {"BRAKE_PRESSED": pressed}
    return self.packer.make_can_msg_panda("BRAKE_MODULE", 0, values)

  def _gas_msg(self, pressed):
    cruise_active = self.safety.get_controls_allowed()
    values = {"GAS_RELEASED": not pressed, "CRUISE_ACTIVE": cruise_active}
    return self.packer.make_can_msg_panda("PCM_CRUISE", 0, values)

  def _pcm_status_msg(self, cruise_on):
    values = {"CRUISE_ACTIVE": cruise_on}
    values["CHECKSUM"] = 1
    return self.packer.make_can_msg_panda("PCM_CRUISE", 0, values)

  def test_prev_gas_interceptor(self):
    self._rx(toyota_interceptor_msg(0x0, 0x201))
    self.assertFalse(self.safety.get_gas_interceptor_prev())
    self._rx(toyota_interceptor_msg(0x1000, 0x201))
    self.assertTrue(self.safety.get_gas_interceptor_prev())
    self._rx(toyota_interceptor_msg(0x0, 0x201))

  def test_disengage_on_gas_interceptor(self):
    for g in range(0, 0x1000):
      self._rx(toyota_interceptor_msg(0, 0x201))
      self.safety.set_controls_allowed(True)
      self._rx(toyota_interceptor_msg(g, 0x201))
      remain_enabled = g <= INTERCEPTOR_THRESHOLD
      self.assertEqual(remain_enabled, self.safety.get_controls_allowed())
      self._rx(toyota_interceptor_msg(0, 0x201))
      self.safety.set_gas_interceptor_detected(False)

  def test_unsafe_mode_no_disengage_on_gas_interceptor(self):
    self.safety.set_controls_allowed(True)
    self.safety.set_unsafe_mode(UNSAFE_MODE.DISABLE_DISENGAGE_ON_GAS)
    for g in range(0, 0x1000):
      self._rx(toyota_interceptor_msg(g, 0x201))
      self.assertTrue(self.safety.get_controls_allowed())
      self._rx(toyota_interceptor_msg(0, 0x201))
      self.safety.set_gas_interceptor_detected(False)
    self.safety.set_unsafe_mode(UNSAFE_MODE.DEFAULT)

  def test_allow_engage_with_gas_interceptor_pressed(self):
    self._rx(toyota_interceptor_msg(0x1000, 0x201))
    self.safety.set_controls_allowed(1)
    self._rx(toyota_interceptor_msg(0x1000, 0x201))
    self.assertTrue(self.safety.get_controls_allowed())
    self._rx(toyota_interceptor_msg(0, 0x201))

  def test_accel_actuation_limits(self):
    limits = ((MIN_ACCEL, MAX_ACCEL, UNSAFE_MODE.DEFAULT),
              (ISO_MIN_ACCEL, ISO_MAX_ACCEL, UNSAFE_MODE.RAISE_LONGITUDINAL_LIMITS_TO_ISO_MAX))

    for min_accel, max_accel, unsafe_mode in limits:
      for accel in np.arange(min_accel - 1, max_accel + 1, 0.1):
        for controls_allowed in [True, False]:
          self.safety.set_controls_allowed(controls_allowed)
          self.safety.set_unsafe_mode(unsafe_mode)
          if controls_allowed:
            should_tx = int(min_accel*1000) <= int(accel*1000) <= int(max_accel*1000)
          else:
            should_tx = np.isclose(accel, 0, atol=0.0001)
          self.assertEqual(should_tx, self._tx(self._accel_msg(accel)))

  def test_torque_absolute_limits(self):
    for controls_allowed in [True, False]:
      for torque in np.arange(-MAX_TORQUE - 1000, MAX_TORQUE + 1000, MAX_RATE_UP):
          self.safety.set_controls_allowed(controls_allowed)
          self.safety.set_toyota_rt_torque_last(torque)
          self.safety.set_toyota_torque_meas(torque, torque)
          self.safety.set_toyota_desired_torque_last(torque - MAX_RATE_UP)

          if controls_allowed:
            send = (-MAX_TORQUE <= torque <= MAX_TORQUE)
          else:
            send = torque == 0

          self.assertEqual(send, self._tx(self._torque_msg(torque)))

  def test_non_realtime_limit_up(self):
    self.safety.set_controls_allowed(True)

    self._set_prev_torque(0)
    self.assertTrue(self._tx(self._torque_msg(MAX_RATE_UP)))

    self._set_prev_torque(0)
    self.assertFalse(self._tx(self._torque_msg(MAX_RATE_UP + 1)))

  def test_non_realtime_limit_down(self):
    self.safety.set_controls_allowed(True)

    self.safety.set_toyota_rt_torque_last(1000)
    self.safety.set_toyota_torque_meas(500, 500)
    self.safety.set_toyota_desired_torque_last(1000)
    self.assertTrue(self._tx(self._torque_msg(1000 - MAX_RATE_DOWN)))

    self.safety.set_toyota_rt_torque_last(1000)
    self.safety.set_toyota_torque_meas(500, 500)
    self.safety.set_toyota_desired_torque_last(1000)
    self.assertFalse(self._tx(self._torque_msg(1000 - MAX_RATE_DOWN + 1)))

  def test_exceed_torque_sensor(self):
    self.safety.set_controls_allowed(True)

    for sign in [-1, 1]:
      self._set_prev_torque(0)
      for t in np.arange(0, MAX_TORQUE_ERROR + 10, 10):
        t *= sign
        self.assertTrue(self._tx(self._torque_msg(t)))

      self.assertFalse(self._tx(self._torque_msg(sign * (MAX_TORQUE_ERROR + 10))))

  def test_realtime_limit_up(self):
    self.safety.set_controls_allowed(True)

    for sign in [-1, 1]:
      self.safety.init_tests_toyota()
      self._set_prev_torque(0)
      for t in np.arange(0, 380, 10):
        t *= sign
        self.safety.set_toyota_torque_meas(t, t)
        self.assertTrue(self._tx(self._torque_msg(t)))
      self.assertFalse(self._tx(self._torque_msg(sign * 380)))

      self._set_prev_torque(0)
      for t in np.arange(0, 370, 10):
        t *= sign
        self.safety.set_toyota_torque_meas(t, t)
        self.assertTrue(self._tx(self._torque_msg(t)))

      # Increase timer to update rt_torque_last
      self.safety.set_timer(RT_INTERVAL + 1)
      self.assertTrue(self._tx(self._torque_msg(sign * 370)))
      self.assertTrue(self._tx(self._torque_msg(sign * 380)))

  def test_torque_measurements(self):
    for trq in [50, -50, 0, 0, 0, 0]:
      self._rx(self._torque_meas_msg(trq))

    # toyota safety adds one to be conservative on rounding
    self.assertEqual(-51, self.safety.get_toyota_torque_meas_min())
    self.assertEqual(51, self.safety.get_toyota_torque_meas_max())

    self._rx(self._torque_meas_msg(0))
    self.assertEqual(1, self.safety.get_toyota_torque_meas_max())
    self.assertEqual(-51, self.safety.get_toyota_torque_meas_min())

    self._rx(self._torque_meas_msg(0))
    self.assertEqual(1, self.safety.get_toyota_torque_meas_max())
    self.assertEqual(-1, self.safety.get_toyota_torque_meas_min())

  def test_gas_interceptor_safety_check(self):
    self.safety.set_controls_allowed(0)
    self.assertTrue(self._tx(toyota_interceptor_msg(0, 0x200)))
    self.assertFalse(self._tx(toyota_interceptor_msg(0x1000, 0x200)))
    self.safety.set_controls_allowed(1)
    self.assertTrue(self._tx(toyota_interceptor_msg(0x1000, 0x200)))

  def test_rx_hook(self):
    # checksum checks
    for msg in ["trq", "pcm"]:
      self.safety.set_controls_allowed(1)
      if msg == "trq":
        to_push = self._torque_meas_msg(0)
      if msg == "pcm":
        to_push = self._pcm_status_msg(True)
      self.assertTrue(self._rx(to_push))
      to_push[0].RDHR = 0
      self.assertFalse(self._rx(to_push))
      self.assertFalse(self.safety.get_controls_allowed())


if __name__ == "__main__":
  unittest.main()
