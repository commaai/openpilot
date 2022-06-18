#!/usr/bin/env python3
import unittest
from panda import Panda
from panda.tests.safety import libpandasafety_py
import panda.tests.safety.common as common
from panda.tests.safety.common import CANPackerPanda


class TestHyundaiHDA2(common.PandaSafetyTest, common.DriverTorqueSteeringSafetyTest):

  TX_MSGS = [[0x50, 0], [0x1CF, 1]]
  STANDSTILL_THRESHOLD = 30  # ~1kph
  RELAY_MALFUNCTION_ADDR = 0x50
  RELAY_MALFUNCTION_BUS = 0
  FWD_BLACKLISTED_ADDRS = {2: [0x50]}
  FWD_BUS_LOOKUP = {0: 2, 2: 0}

  MAX_RATE_UP = 3
  MAX_RATE_DOWN = 7
  MAX_TORQUE = 150

  MAX_RT_DELTA = 112
  RT_INTERVAL = 250000

  DRIVER_TORQUE_ALLOWANCE = 50
  DRIVER_TORQUE_FACTOR = 2

  def setUp(self):
    self.packer = CANPackerPanda("kia_ev6")
    self.safety = libpandasafety_py.libpandasafety
    self.safety.set_safety_hooks(Panda.SAFETY_HYUNDAI_HDA2, 0)
    self.safety.init_tests()

  def _torque_driver_msg(self, torque):
    values = {"STEERING_COL_TORQUE": torque}
    return self.packer.make_can_msg_panda("MDPS", 1, values, counter=True)

  def _torque_cmd_msg(self, torque, steer_req=1):
    values = {"TORQUE_REQUEST": torque}
    return self.packer.make_can_msg_panda("LKAS", 0, values, counter=True)

  def _speed_msg(self, speed):
    values = {f"WHEEL_SPEED_{i}": speed * 0.03125 for i in range(1, 5)}
    return self.packer.make_can_msg_panda("WHEEL_SPEEDS", 1, values, counter=True)

  def _user_brake_msg(self, brake):
    values = {"BRAKE_PRESSED": brake}
    return self.packer.make_can_msg_panda("BRAKE", 1, values, counter=True)

  def _user_gas_msg(self, gas):
    values = {"ACCELERATOR_PEDAL": gas}
    return self.packer.make_can_msg_panda("ACCELERATOR", 1, values, counter=True)

  def _pcm_status_msg(self, enable):
    values = {"CRUISE_ACTIVE": enable}
    return self.packer.make_can_msg_panda("SCC1", 1, values, counter=True)

  def _button_msg(self, resume=False, cancel=False):
    values = {
      "DISTANCE_BTN": resume,
      "PAUSE_RESUME_BTN": cancel,
    }
    return self.packer.make_can_msg_panda("CRUISE_BUTTONS", 1, values)

  def test_buttons(self):
    for controls_allowed in (True, False):
      for cruise_engaged in (True, False):
        self._rx(self._pcm_status_msg(cruise_engaged))
        self.safety.set_controls_allowed(controls_allowed)

        self.assertEqual(cruise_engaged, self._tx(self._button_msg(cancel=True)))
        self.assertEqual(controls_allowed, self._tx(self._button_msg(resume=True)))


if __name__ == "__main__":
  unittest.main()
