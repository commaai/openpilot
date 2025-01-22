#!/usr/bin/env python3
import unittest
from panda import Panda
from panda.tests.libpanda import libpanda_py
import panda.tests.safety.common as common
from panda.tests.safety.common import CANPackerPanda


class TestNissanSafety(common.PandaCarSafetyTest, common.AngleSteeringSafetyTest):

  TX_MSGS = [[0x169, 0], [0x2b1, 0], [0x4cc, 0], [0x20b, 2], [0x280, 2]]
  STANDSTILL_THRESHOLD = 0
  GAS_PRESSED_THRESHOLD = 3
  RELAY_MALFUNCTION_ADDRS = {0: (0x169,)}
  FWD_BLACKLISTED_ADDRS = {0: [0x280], 2: [0x169, 0x2b1, 0x4cc]}
  FWD_BUS_LOOKUP = {0: 2, 2: 0}

  EPS_BUS = 0
  CRUISE_BUS = 2

  # Angle control limits
  DEG_TO_CAN = 100

  ANGLE_RATE_BP = [0., 5., 15.]
  ANGLE_RATE_UP = [5., .8, .15]  # windup limit
  ANGLE_RATE_DOWN = [5., 3.5, .4]  # unwind limit

  def setUp(self):
    self.packer = CANPackerPanda("nissan_x_trail_2017_generated")
    self.safety = libpanda_py.libpanda
    self.safety.set_safety_hooks(Panda.SAFETY_NISSAN, 0)
    self.safety.init_tests()

  def _angle_cmd_msg(self, angle: float, enabled: bool):
    values = {"DESIRED_ANGLE": angle, "LKA_ACTIVE": 1 if enabled else 0}
    return self.packer.make_can_msg_panda("LKAS", 0, values)

  def _angle_meas_msg(self, angle: float):
    values = {"STEER_ANGLE": angle}
    return self.packer.make_can_msg_panda("STEER_ANGLE_SENSOR", self.EPS_BUS, values)

  def _pcm_status_msg(self, enable):
    values = {"CRUISE_ENABLED": enable}
    return self.packer.make_can_msg_panda("CRUISE_STATE", self.CRUISE_BUS, values)

  def _speed_msg(self, speed):
    values = {"WHEEL_SPEED_%s" % s: speed * 3.6 for s in ["RR", "RL"]}
    return self.packer.make_can_msg_panda("WHEEL_SPEEDS_REAR", self.EPS_BUS, values)

  def _user_brake_msg(self, brake):
    values = {"USER_BRAKE_PRESSED": brake}
    return self.packer.make_can_msg_panda("DOORS_LIGHTS", self.EPS_BUS, values)

  def _user_gas_msg(self, gas):
    values = {"GAS_PEDAL": gas}
    return self.packer.make_can_msg_panda("GAS_PEDAL", self.EPS_BUS, values)

  def _acc_button_cmd(self, cancel=0, propilot=0, flw_dist=0, _set=0, res=0):
    no_button = not any([cancel, propilot, flw_dist, _set, res])
    values = {"CANCEL_BUTTON": cancel, "PROPILOT_BUTTON": propilot,
              "FOLLOW_DISTANCE_BUTTON": flw_dist, "SET_BUTTON": _set,
              "RES_BUTTON": res, "NO_BUTTON_PRESSED": no_button}
    return self.packer.make_can_msg_panda("CRUISE_THROTTLE", 2, values)

  def test_acc_buttons(self):
    btns = [
      ("cancel", True),
      ("propilot", False),
      ("flw_dist", False),
      ("_set", False),
      ("res", False),
      (None, False),
    ]
    for controls_allowed in (True, False):
      for btn, should_tx in btns:
        self.safety.set_controls_allowed(controls_allowed)
        args = {} if btn is None else {btn: 1}
        tx = self._tx(self._acc_button_cmd(**args))
        self.assertEqual(tx, should_tx)


class TestNissanSafetyAltEpsBus(TestNissanSafety):
  """Altima uses different buses"""

  EPS_BUS = 1
  CRUISE_BUS = 1

  def setUp(self):
    self.packer = CANPackerPanda("nissan_x_trail_2017_generated")
    self.safety = libpanda_py.libpanda
    self.safety.set_safety_hooks(Panda.SAFETY_NISSAN, Panda.FLAG_NISSAN_ALT_EPS_BUS)
    self.safety.init_tests()


class TestNissanLeafSafety(TestNissanSafety):

  def setUp(self):
    self.packer = CANPackerPanda("nissan_leaf_2018_generated")
    self.safety = libpanda_py.libpanda
    self.safety.set_safety_hooks(Panda.SAFETY_NISSAN, 0)
    self.safety.init_tests()

  def _user_brake_msg(self, brake):
    values = {"USER_BRAKE_PRESSED": brake}
    return self.packer.make_can_msg_panda("CRUISE_THROTTLE", 0, values)

  def _user_gas_msg(self, gas):
    values = {"GAS_PEDAL": gas}
    return self.packer.make_can_msg_panda("CRUISE_THROTTLE", 0, values)

  # TODO: leaf should use its own safety param
  def test_acc_buttons(self):
    pass


if __name__ == "__main__":
  unittest.main()
