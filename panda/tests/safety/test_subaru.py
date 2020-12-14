#!/usr/bin/env python3
import unittest
import numpy as np
from panda import Panda
from panda.tests.safety import libpandasafety_py
import panda.tests.safety.common as common
from panda.tests.safety.common import CANPackerPanda

MAX_RATE_UP = 50
MAX_RATE_DOWN = 70
MAX_STEER = 2047

MAX_RT_DELTA = 940
RT_INTERVAL = 250000

DRIVER_TORQUE_ALLOWANCE = 60
DRIVER_TORQUE_FACTOR = 10


class TestSubaruSafety(common.PandaSafetyTest):
  cnt_gas = 0
  cnt_torque_driver = 0
  cnt_cruise = 0
  cnt_speed = 0
  cnt_brake = 0

  TX_MSGS = [[0x122, 0], [0x221, 0], [0x322, 0]]
  STANDSTILL_THRESHOLD = 20  # 1kph (see dbc file)
  RELAY_MALFUNCTION_ADDR = 0x122
  RELAY_MALFUNCTION_BUS = 0
  FWD_BLACKLISTED_ADDRS = {2: [0x122, 0x221, 0x322]}
  FWD_BUS_LOOKUP = {0: 2, 2: 0}

  def setUp(self):
    self.packer = CANPackerPanda("subaru_global_2017_generated")
    self.safety = libpandasafety_py.libpandasafety
    self.safety.set_safety_hooks(Panda.SAFETY_SUBARU, 0)
    self.safety.init_tests()

  def _set_prev_torque(self, t):
    self.safety.set_desired_torque_last(t)
    self.safety.set_rt_torque_last(t)

  def _torque_driver_msg(self, torque):
    values = {"Steer_Torque_Sensor": torque, "Counter": self.cnt_torque_driver % 4}
    self.__class__.cnt_torque_driver += 1
    return self.packer.make_can_msg_panda("Steering_Torque", 0, values)

  def _speed_msg(self, speed):
    # subaru safety doesn't use the scaled value, so undo the scaling
    values = {s: speed * 0.057 for s in ["FR", "FL", "RR", "RL"]}
    values["Counter"] = self.cnt_speed % 4
    self.__class__.cnt_speed += 1
    return self.packer.make_can_msg_panda("Wheel_Speeds", 0, values)

  def _brake_msg(self, brake):
    values = {"Brake_Pedal": brake, "Counter": self.cnt_brake % 4}
    self.__class__.cnt_brake += 1
    return self.packer.make_can_msg_panda("Brake_Pedal", 0, values)

  def _torque_msg(self, torque):
    values = {"LKAS_Output": torque}
    return self.packer.make_can_msg_panda("ES_LKAS", 0, values)

  def _gas_msg(self, gas):
    values = {"Throttle_Pedal": gas, "Counter": self.cnt_gas % 4}
    self.__class__.cnt_gas += 1
    return self.packer.make_can_msg_panda("Throttle", 0, values)

  def _pcm_status_msg(self, enable):
    values = {"Cruise_Activated": enable, "Counter": self.cnt_cruise % 4}
    self.__class__.cnt_cruise += 1
    return self.packer.make_can_msg_panda("CruiseControl", 0, values)

  def _set_torque_driver(self, min_t, max_t):
    for _ in range(0, 5):
      self._rx(self._torque_driver_msg(min_t))
    self._rx(self._torque_driver_msg(max_t))

  def test_steer_safety_check(self):
    for enabled in [0, 1]:
      for t in range(-3000, 3000):
        self.safety.set_controls_allowed(enabled)
        self._set_prev_torque(t)
        block = abs(t) > MAX_STEER or (not enabled and abs(t) > 0)
        self.assertEqual(not block, self._tx(self._torque_msg(t)))

  def test_non_realtime_limit_up(self):
    self._set_torque_driver(0, 0)
    self.safety.set_controls_allowed(True)

    self._set_prev_torque(0)
    self.assertTrue(self._tx(self._torque_msg(MAX_RATE_UP)))
    self._set_prev_torque(0)
    self.assertTrue(self._tx(self._torque_msg(-MAX_RATE_UP)))

    self._set_prev_torque(0)
    self.assertFalse(self._tx(self._torque_msg(MAX_RATE_UP + 1)))
    self.safety.set_controls_allowed(True)
    self._set_prev_torque(0)
    self.assertFalse(self._tx(self._torque_msg(-MAX_RATE_UP - 1)))

  def test_non_realtime_limit_down(self):
    self._set_torque_driver(0, 0)
    self.safety.set_controls_allowed(True)

  def test_against_torque_driver(self):
    self.safety.set_controls_allowed(True)

    for sign in [-1, 1]:
      for t in np.arange(0, DRIVER_TORQUE_ALLOWANCE + 1, 1):
        t *= -sign
        self._set_torque_driver(t, t)
        self._set_prev_torque(MAX_STEER * sign)
        self.assertTrue(self._tx(self._torque_msg(MAX_STEER * sign)))

      self._set_torque_driver(DRIVER_TORQUE_ALLOWANCE + 1, DRIVER_TORQUE_ALLOWANCE + 1)
      self.assertFalse(self._tx(self._torque_msg(-MAX_STEER)))

    # arbitrary high driver torque to ensure max steer torque is allowed
    max_driver_torque = int(MAX_STEER / DRIVER_TORQUE_FACTOR + DRIVER_TORQUE_ALLOWANCE + 1)

    # spot check some individual cases
    for sign in [-1, 1]:
      driver_torque = (DRIVER_TORQUE_ALLOWANCE + 10) * sign
      torque_desired = (MAX_STEER - 10 * DRIVER_TORQUE_FACTOR) * sign
      delta = 1 * sign
      self._set_prev_torque(torque_desired)
      self._set_torque_driver(-driver_torque, -driver_torque)
      self.assertTrue(self._tx(self._torque_msg(torque_desired)))
      self._set_prev_torque(torque_desired + delta)
      self._set_torque_driver(-driver_torque, -driver_torque)
      self.assertFalse(self._tx(self._torque_msg(torque_desired + delta)))

      self._set_prev_torque(MAX_STEER * sign)
      self._set_torque_driver(-max_driver_torque * sign, -max_driver_torque * sign)
      self.assertTrue(self._tx(self._torque_msg((MAX_STEER - MAX_RATE_DOWN) * sign)))
      self._set_prev_torque(MAX_STEER * sign)
      self._set_torque_driver(-max_driver_torque * sign, -max_driver_torque * sign)
      self.assertTrue(self._tx(self._torque_msg(0)))
      self._set_prev_torque(MAX_STEER * sign)
      self._set_torque_driver(-max_driver_torque * sign, -max_driver_torque * sign)
      self.assertFalse(self._tx(self._torque_msg((MAX_STEER - MAX_RATE_DOWN + 1) * sign)))

  def test_realtime_limits(self):
    self.safety.set_controls_allowed(True)

    for sign in [-1, 1]:
      self.safety.init_tests()
      self._set_prev_torque(0)
      self._set_torque_driver(0, 0)
      for t in np.arange(0, MAX_RT_DELTA, 1):
        t *= sign
        self.assertTrue(self._tx(self._torque_msg(t)))
      self.assertFalse(self._tx(self._torque_msg(sign * (MAX_RT_DELTA + 1))))

      self._set_prev_torque(0)
      for t in np.arange(0, MAX_RT_DELTA, 1):
        t *= sign
        self.assertTrue(self._tx(self._torque_msg(t)))

      # Increase timer to update rt_torque_last
      self.safety.set_timer(RT_INTERVAL + 1)
      self.assertTrue(self._tx(self._torque_msg(sign * (MAX_RT_DELTA - 1))))
      self.assertTrue(self._tx(self._torque_msg(sign * (MAX_RT_DELTA + 1))))


if __name__ == "__main__":
  unittest.main()
