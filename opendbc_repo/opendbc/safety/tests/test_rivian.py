#!/usr/bin/env python3
import unittest

from opendbc.car.structs import CarParams
from opendbc.safety.tests.libsafety import libsafety_py
import opendbc.safety.tests.common as common
from opendbc.safety.tests.common import CANPackerPanda
from opendbc.car.rivian.values import RivianSafetyFlags


class TestRivianSafetyBase(common.PandaCarSafetyTest, common.DriverTorqueSteeringSafetyTest):

  TX_MSGS = [[0x120, 0], [0x321, 2]]
  STANDSTILL_THRESHOLD = 0
  RELAY_MALFUNCTION_ADDRS = {0: (0x120,)}
  FWD_BLACKLISTED_ADDRS = {0: [0x321], 2: [0x120]}
  FWD_BUS_LOOKUP = {0: 2, 2: 0}

  MAX_TORQUE = 250
  MAX_RATE_UP = 3
  MAX_RATE_DOWN = 5

  MAX_RT_DELTA = 125
  RT_INTERVAL = 250000

  DRIVER_TORQUE_ALLOWANCE = 100
  DRIVER_TORQUE_FACTOR = 2

  @classmethod
  def setUpClass(cls):
    if cls.__name__ == "TestRivianSafetyBase":
      raise unittest.SkipTest

  def _torque_driver_msg(self, torque):
    values = {"EPAS_TorsionBarTorque": torque / 100.0}
    return self.packer.make_can_msg_panda("EPAS_SystemStatus", 0, values)

  def _torque_cmd_msg(self, torque, steer_req=1):
    values = {"ACM_lkaStrToqReq": torque, "ACM_lkaActToi": steer_req}
    return self.packer.make_can_msg_panda("ACM_lkaHbaCmd", 0, values)

  def _speed_msg(self, speed):
    values = {"ESP_Status": speed * 3.6}
    return self.packer.make_can_msg_panda("ESP_Vehicle_Speed", 0, values)

  def _user_brake_msg(self, brake):
    values = {"iBESP2_BrakePedalApplied": brake}
    return self.packer.make_can_msg_panda("iBESP2", 0, values)

  def _user_gas_msg(self, gas):
    values = {"VDM_AcceleratorPedalPosition": gas}
    return self.packer.make_can_msg_panda("VDM_PropStatus", 0, values)

  def _pcm_status_msg(self, enable):
    values = {"ACM_FeatureStatus": enable, "ACM_Unkown1": 1}
    return self.packer.make_can_msg_panda("ACM_Status", 2, values)

  def _vehicle_moving_msg(self, speed: float):
    values = {"ESP_Vehicle_Speed": speed}
    return self.packer.make_can_msg_panda("ESP_Status", 0, values)

  def _accel_msg(self, accel: float):
    values = {"ACM_AccelerationRequest": accel}
    return self.packer.make_can_msg_panda("ACM_longitudinalRequest", 0, values)

  def test_wheel_touch(self):
    self.safety.set_controls_allowed(True)
    for controls_allowed in (True, False):
      self.safety.set_controls_allowed(controls_allowed)
      values = {
        "SCCM_WheelTouch_HandsOn": 1 if controls_allowed else 0,
        "SCCM_WheelTouch_CapacitiveValue": 100 if controls_allowed else 0,
        "SETME_X52": 100,
      }
      self.assertTrue(self._tx(self.packer.make_can_msg_panda("SCCM_WheelTouch", 2, values)))


class TestRivianStockSafety(TestRivianSafetyBase):

  def setUp(self):
    self.packer = CANPackerPanda("rivian_primary_actuator")
    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(CarParams.SafetyModel.rivian, 0)
    self.safety.init_tests()


class TestRivianLongitudinalSafety(TestRivianSafetyBase, common.LongitudinalAccelSafetyTest):
  RELAY_MALFUNCTION_ADDRS = {0: (0x120, 0x160)}
  FWD_BLACKLISTED_ADDRS = {0: [0x321], 2: [0x120, 0x160]}

  LONGITUDINAL = True

  def setUp(self):
    self.packer = CANPackerPanda("rivian_primary_actuator")
    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(CarParams.SafetyModel.rivian, RivianSafetyFlags.LONG_CONTROL)
    self.safety.init_tests()


if __name__ == "__main__":
  unittest.main()
