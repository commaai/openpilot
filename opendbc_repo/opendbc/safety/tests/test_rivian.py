#!/usr/bin/env python3
import unittest

from opendbc.car.structs import CarParams
import opendbc.safety.tests.common as common
from opendbc.car.rivian.values import RivianSafetyFlags


class TestRivianSafetyBase(common.CarSafetyTest, common.DriverTorqueSteeringSafetyTest, common.LongitudinalAccelSafetyTest,
                           common.VehicleSpeedSafetyTest):

  DBC = "rivian_primary_actuator"
  SAFETY_MODEL = CarParams.SafetyModel.rivian

  TX_MSGS = [[0x120, 0], [0x321, 2], [0x162, 2]]
  RELAY_MALFUNCTION_ADDRS = {0: (0x120,), 2: (0x321, 0x162)}
  FWD_BLACKLISTED_ADDRS = {0: [0x321, 0x162], 2: [0x120]}

  MAX_TORQUE_LOOKUP = [9, 17], [350, 250]
  DYNAMIC_MAX_TORQUE = True
  MAX_RATE_UP = 3
  MAX_RATE_DOWN = 5

  MAX_RT_DELTA = 125

  DRIVER_TORQUE_ALLOWANCE = 100
  DRIVER_TORQUE_FACTOR = 2

  cnt_speed = 0
  cnt_speed_2 = 0
  cnt_torque = 0
  cnt_brake = 0
  cnt_pcm = 0

  RX_QUALITY_FLAGS = {
    "ESP_Status": {"ESP_Vehicle_Speed_Q": (1,)},
    "VDM_PropStatus": {"VDM_VehicleSpeedQ": (1,)},
    "iBESP2": {"iBESP2_BrakePedalApplied_Q": (1,)},
  }

  def _torque_driver_msg(self, torque):
    values = {"EPAS_TorsionBarTorque": torque / 100.0, "EPAS_SystemStatus_Counter": self.cnt_torque % 15}
    self.__class__.cnt_torque += 1
    return self.packer.make_can_msg_safety("EPAS_SystemStatus", 0, values)

  def _torque_cmd_msg(self, torque, steer_req=1):
    values = {"ACM_lkaStrToqReq": torque, "ACM_lkaActToi": steer_req}
    return self.packer.make_can_msg_safety("ACM_lkaHbaCmd", 0, values)

  def _speed_msg(self, speed):
    values = {"ESP_Vehicle_Speed": speed * 3.6, "ESP_Status_Counter": self.cnt_speed % 15, "ESP_Vehicle_Speed_Q": 1}
    self.__class__.cnt_speed += 1
    return self.packer.make_can_msg_safety("ESP_Status", 0, values)

  def _speed_msg_2(self, speed):
    # Rivian has a dynamic max torque limit based on speed, so it checks two sources
    return self._user_gas_msg(0, speed)

  def _user_brake_msg(self, brake):
    values = {"iBESP2_BrakePedalApplied": brake, "iBESP2_BrakePedalApplied_Q": 1,
              "iBESP2_AliveCounter": self.cnt_brake % 15}
    self.__class__.cnt_brake += 1
    return self.packer.make_can_msg_safety("iBESP2", 0, values)

  def _user_gas_msg(self, gas, speed=0):
    values = {"VDM_AcceleratorPedalPosition": gas, "VDM_VehicleSpeed": speed * 3.6,
              "VDM_PropStatus_Counter": self.cnt_speed_2 % 15, "VDM_VehicleSpeedQ": 1}
    self.__class__.cnt_speed_2 += 1
    return self.packer.make_can_msg_safety("VDM_PropStatus", 0, values)

  def _pcm_status_msg(self, enable):
    values = {"ACM_FeatureStatus": enable, "ACM_Unkown1": 1, "ACM_Status_Counter": self.cnt_pcm % 15}
    self.__class__.cnt_pcm += 1
    return self.packer.make_can_msg_safety("ACM_Status", 2, values)

  def _accel_msg(self, accel: float):
    values = {"ACM_AccelerationRequest": accel}
    return self.packer.make_can_msg_safety("ACM_longitudinalRequest", 0, values)

  def test_wheel_touch(self):
    # For hiding hold wheel alert on engage
    for controls_allowed in (True, False):
      self.safety.set_controls_allowed(controls_allowed)
      values = {
        "SCCM_WheelTouch_HandsOn": 1 if controls_allowed else 0,
        "SCCM_WheelTouch_CapacitiveValue": 100 if controls_allowed else 0,
        "SCCM_WheelTouch_Calibration": 100,
      }
      self.assertTrue(self._tx(self.packer.make_can_msg_safety("SCCM_WheelTouch", 2, values)))


class TestRivianStockSafety(TestRivianSafetyBase):

  LONGITUDINAL = False

  def test_adas_status(self):
    # For canceling stock ACC
    for controls_allowed in (True, False):
      self.safety.set_controls_allowed(controls_allowed)
      for interface_status in range(4):
        values = {"VDM_AdasInterfaceStatus": interface_status}
        self.assertTrue(self._tx(self.packer.make_can_msg_safety("VDM_AdasSts", 2, values)))


class TestRivianLongitudinalSafety(TestRivianSafetyBase):

  SAFETY_PARAM = RivianSafetyFlags.LONG_CONTROL

  TX_MSGS = [[0x120, 0], [0x321, 2], [0x160, 0]]
  RELAY_MALFUNCTION_ADDRS = {0: (0x120, 0x160), 2: (0x321,)}
  FWD_BLACKLISTED_ADDRS = {0: [0x321], 2: [0x120, 0x160]}


class TestRivianIgnition(common.CanIgnitionSafetyTest):
  DBC = "rivian_primary_actuator"
  IGNITION_VALUES = range(4)
  IGNITION_ON_VALUES = {1}  # VDM_EpasPowerMode_Drive_On
  IGNITION_COUNTER_MOD = 15

  def _ignition_msg(self, value, counter=None, bus=0):
    return self.packer.make_can_msg_safety("VDM_OutputSignals", bus,
                                           {"VDM_OutputSigs_Counter": counter, "VDM_EpasPowerMode": value})


if __name__ == "__main__":
  unittest.main()
