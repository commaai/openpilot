#!/usr/bin/env python3
import unittest
from panda import Panda
from panda.tests.safety import libpandasafety_py
import panda.tests.safety.common as common
from panda.tests.safety.common import CANPackerPanda


class TestChryslerSafety(common.PandaSafetyTest, common.MotorTorqueSteeringSafetyTest):
  TX_MSGS = [[571, 0], [658, 0], [678, 0]]
  STANDSTILL_THRESHOLD = 0
  RELAY_MALFUNCTION_ADDR = 0x292
  RELAY_MALFUNCTION_BUS = 0
  FWD_BLACKLISTED_ADDRS = {2: [658, 678]}
  FWD_BUS_LOOKUP = {0: 2, 2: 0}

  MAX_RATE_UP = 3
  MAX_RATE_DOWN = 3
  MAX_TORQUE = 261
  MAX_RT_DELTA = 112
  RT_INTERVAL = 250000
  MAX_TORQUE_ERROR = 80

  cnt_torque_meas = 0
  cnt_gas = 0
  cnt_cruise = 0
  cnt_brake = 0

  def setUp(self):
    self.packer = CANPackerPanda("chrysler_pacifica_2017_hybrid")
    self.safety = libpandasafety_py.libpandasafety
    self.safety.set_safety_hooks(Panda.SAFETY_CHRYSLER, 0)
    self.safety.init_tests()

  def _button_msg(self, cancel):
    values = {"ACC_CANCEL": cancel}
    return self.packer.make_can_msg_panda("WHEEL_BUTTONS", 0, values)

  def _pcm_status_msg(self, enable):
    values = {"ACC_STATUS_2": 0x7 if enable else 0,
              "COUNTER": self.cnt_cruise % 16}
    self.__class__.cnt_cruise += 1
    return self.packer.make_can_msg_panda("ACC_2", 0, values)

  def _speed_msg(self, speed):
    values = {"SPEED_LEFT": speed, "SPEED_RIGHT": speed}
    return self.packer.make_can_msg_panda("SPEED_1", 0, values)

  def _user_gas_msg(self, gas):
    values = {"Accelerator_Position": gas, "COUNTER": self.cnt_gas % 16}
    self.__class__.cnt_gas += 1
    return self.packer.make_can_msg_panda("ECM_5", 0, values)

  def _user_brake_msg(self, brake):
    values = {"Brake_Pedal_State": 1 if brake else 0,
              "COUNTER": self.cnt_brake % 16}
    self.__class__.cnt_brake += 1
    return self.packer.make_can_msg_panda("ESP_1", 0, values)

  def _torque_meas_msg(self, torque):
    values = {"TORQUE_MOTOR": torque, "COUNTER": self.cnt_torque_meas % 16}
    self.__class__.cnt_torque_meas += 1
    return self.packer.make_can_msg_panda("EPS_STATUS", 0, values)

  def _torque_cmd_msg(self, torque, steer_req=1):
    values = {"LKAS_STEERING_TORQUE": torque}
    return self.packer.make_can_msg_panda("LKAS_COMMAND", 0, values)

  def test_cancel_button(self):
    for cancel in [True, False]:
      self.assertEqual(cancel, self._tx(self._button_msg(cancel)))


if __name__ == "__main__":
  unittest.main()
