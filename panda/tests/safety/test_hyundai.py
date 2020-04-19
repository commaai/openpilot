#!/usr/bin/env python3
import unittest
import numpy as np
from panda import Panda
from panda.tests.safety import libpandasafety_py
import panda.tests.safety.common as common
from panda.tests.safety.common import CANPackerPanda

MAX_RATE_UP = 3
MAX_RATE_DOWN = 7
MAX_STEER = 255

MAX_RT_DELTA = 112
RT_INTERVAL = 250000

DRIVER_TORQUE_ALLOWANCE = 50
DRIVER_TORQUE_FACTOR = 2


class TestHyundaiSafety(common.PandaSafetyTest):
  TX_MSGS = [[832, 0], [1265, 0], [1157, 0]]
  STANDSTILL_THRESHOLD = 30  # ~1kph
  RELAY_MALFUNCTION_ADDR = 832
  RELAY_MALFUNCTION_BUS = 0
  FWD_BLACKLISTED_ADDRS = {2: [832, 1157]}
  FWD_BUS_LOOKUP = {0: 2, 2: 0}

  def setUp(self):
    self.packer = CANPackerPanda("hyundai_kia_generic")
    self.safety = libpandasafety_py.libpandasafety
    self.safety.set_safety_hooks(Panda.SAFETY_HYUNDAI, 0)
    self.safety.init_tests_hyundai()

  def _button_msg(self, buttons):
    values = {"CF_Clu_CruiseSwState": buttons}
    return self.packer.make_can_msg_panda("CLU11", 0, values)

  def _gas_msg(self, val):
    values = {"CF_Ems_AclAct": val}
    return self.packer.make_can_msg_panda("EMS16", 0, values)

  def _brake_msg(self, brake):
    values = {"DriverBraking": brake}
    return self.packer.make_can_msg_panda("TCS13", 0, values)

  def _speed_msg(self, speed):
    # panda safety doesn't scale, so undo the scaling
    values = {"WHL_SPD_%s"%s: speed*0.03125 for s in ["FL", "FR", "RL", "RR"]}
    return self.packer.make_can_msg_panda("WHL_SPD11", 0, values)

  def _pcm_status_msg(self, enabled):
    values = {"ACCMode": enabled}
    return self.packer.make_can_msg_panda("SCC12", 0, values)

  def _set_prev_torque(self, t):
    self.safety.set_hyundai_desired_torque_last(t)
    self.safety.set_hyundai_rt_torque_last(t)

  # TODO: this is unused
  def _torque_driver_msg(self, torque):
    values = {"CR_Mdps_StrColTq": torque}
    return self.packer.make_can_msg_panda("MDPS12", 0, values)

  def _torque_msg(self, torque):
    values = {"CR_Lkas_StrToqReq": torque}
    return self.packer.make_can_msg_panda("LKAS11", 0, values)

  def test_steer_safety_check(self):
    for enabled in [0, 1]:
      for t in range(-0x200, 0x200):
        self.safety.set_controls_allowed(enabled)
        self._set_prev_torque(t)
        if abs(t) > MAX_STEER or (not enabled and abs(t) > 0):
          self.assertFalse(self.safety.safety_tx_hook(self._torque_msg(t)))
        else:
          self.assertTrue(self.safety.safety_tx_hook(self._torque_msg(t)))

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


  def test_spam_cancel_safety_check(self):
    RESUME_BTN = 1
    SET_BTN = 2
    CANCEL_BTN = 4
    self.safety.set_controls_allowed(0)
    self.assertTrue(self.safety.safety_tx_hook(self._button_msg(CANCEL_BTN)))
    self.assertFalse(self.safety.safety_tx_hook(self._button_msg(RESUME_BTN)))
    self.assertFalse(self.safety.safety_tx_hook(self._button_msg(SET_BTN)))
    # do not block resume if we are engaged already
    self.safety.set_controls_allowed(1)
    self.assertTrue(self.safety.safety_tx_hook(self._button_msg(RESUME_BTN)))


if __name__ == "__main__":
  unittest.main()
