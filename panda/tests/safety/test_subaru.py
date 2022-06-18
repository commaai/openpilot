#!/usr/bin/env python3
import unittest
from panda import Panda
from panda.tests.safety import libpandasafety_py
import panda.tests.safety.common as common
from panda.tests.safety.common import CANPackerPanda


class TestSubaruSafety(common.PandaSafetyTest, common.DriverTorqueSteeringSafetyTest):
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

  MAX_RATE_UP = 50
  MAX_RATE_DOWN = 70
  MAX_TORQUE = 2047

  MAX_RT_DELTA = 940
  RT_INTERVAL = 250000

  DRIVER_TORQUE_ALLOWANCE = 60
  DRIVER_TORQUE_FACTOR = 10

  def setUp(self):
    self.packer = CANPackerPanda("subaru_global_2017_generated")
    self.safety = libpandasafety_py.libpandasafety
    self.safety.set_safety_hooks(Panda.SAFETY_SUBARU, 0)
    self.safety.init_tests()

  def _set_prev_torque(self, t):
    self.safety.set_desired_torque_last(t)
    self.safety.set_rt_torque_last(t)

  # TODO: this is unused
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

  def _user_brake_msg(self, brake):
    values = {"Brake": brake, "Counter": self.cnt_brake % 4}
    self.__class__.cnt_brake += 1
    return self.packer.make_can_msg_panda("Brake_Status", 0, values)

  def _torque_cmd_msg(self, torque, steer_req=1):
    values = {"LKAS_Output": torque}
    return self.packer.make_can_msg_panda("ES_LKAS", 0, values)

  def _user_gas_msg(self, gas):
    values = {"Throttle_Pedal": gas, "Counter": self.cnt_gas % 4}
    self.__class__.cnt_gas += 1
    return self.packer.make_can_msg_panda("Throttle", 0, values)

  def _pcm_status_msg(self, enable):
    values = {"Cruise_Activated": enable, "Counter": self.cnt_cruise % 4}
    self.__class__.cnt_cruise += 1
    return self.packer.make_can_msg_panda("CruiseControl", 0, values)


if __name__ == "__main__":
  unittest.main()
