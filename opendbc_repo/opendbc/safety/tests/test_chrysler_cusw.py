#!/usr/bin/env python3
import unittest
from opendbc.car.structs import CarParams
from opendbc.safety.tests.libsafety import libsafety_py
import opendbc.safety.tests.common as common
from opendbc.safety.tests.common import CANPackerSafety


class TestChryslerCuswSafety(common.CarSafetyTest, common.MotorTorqueSteeringSafetyTest):
  TX_MSGS = [[0x1F6, 0], [0x2FA, 0], [0x5DC, 0]]
  STANDSTILL_THRESHOLD = 0
  RELAY_MALFUNCTION_ADDRS = {0: (0x1F6, 0x5DC)}
  FWD_BLACKLISTED_ADDRS = {2: [0x1F6, 0x5DC]}

  MAX_RATE_UP = 4
  MAX_RATE_DOWN = 4
  MAX_TORQUE_LOOKUP = [0], [250]
  MAX_RT_DELTA = 150
  MAX_TORQUE_ERROR = 80

  def setUp(self):
    self.packer = CANPackerSafety("chrysler_cusw")
    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(CarParams.SafetyModel.chryslerCusw, 0)
    self.safety.init_tests()

  def _button_msg(self, cancel=False, resume=False):
    values = {"ACC_Cancel": cancel, "ACC_Resume": resume}
    return self.packer.make_can_msg_safety("CRUISE_BUTTONS", 0, values)

  def _pcm_status_msg(self, enable):
    values = {"ACC_ACTIVE": 1 if enable else 0}
    return self.packer.make_can_msg_safety("ACC_CONTROL", 0, values)

  def _speed_msg(self, speed):
    values = {"VEHICLE_SPEED": speed}
    return self.packer.make_can_msg_safety("BRAKE_1", 0, values)

  def _user_gas_msg(self, gas):
    values = {"GAS_HUMAN": gas}
    return self.packer.make_can_msg_safety("ACCEL_GAS", 0, values)

  def _user_brake_msg(self, brake):
    values = {"DRIVER_BRAKE_SWITCH": 1 if brake else 0}
    return self.packer.make_can_msg_safety("BRAKE_3", 0, values)

  def _torque_meas_msg(self, torque):
    values = {"TORQUE_MOTOR": torque}
    return self.packer.make_can_msg_safety("EPS_STATUS", 0, values)

  def _torque_cmd_msg(self, torque, steer_req=1):
    values = {"STEERING_TORQUE": torque, "LKAS_CONTROL_BIT": steer_req}
    return self.packer.make_can_msg_safety("LKAS_COMMAND", 0, values)

  def test_buttons(self):
    for controls_allowed in (True, False):
      self.safety.set_controls_allowed(controls_allowed)

      # resume only while controls allowed
      self.assertEqual(controls_allowed, self._tx(self._button_msg(resume=True)))

      # can always cancel
      self.assertTrue(self._tx(self._button_msg(cancel=True)))

  def test_rx_hook(self):
    for count in range(20):
      self.assertTrue(self._rx(self._speed_msg(0)), f"{count=}")
      self.assertTrue(self._rx(self._user_brake_msg(False)), f"{count=}")
      self.assertTrue(self._rx(self._torque_meas_msg(0)), f"{count=}")
      self.assertTrue(self._rx(self._user_gas_msg(0)), f"{count=}")
      self.assertTrue(self._rx(self._pcm_status_msg(False)), f"{count=}")


if __name__ == "__main__":
  unittest.main()
