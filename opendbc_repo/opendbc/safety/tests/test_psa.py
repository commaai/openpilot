#!/usr/bin/env python3
import unittest

from opendbc.car.structs import CarParams
import opendbc.safety.tests.common as common

LANE_KEEP_ASSIST = 0x3F2


class TestPsaSafetyBase(common.CarSafetyTest, common.AngleSteeringSafetyTest):
  DBC = "psa_aee2010_r3"
  SAFETY_MODEL = CarParams.SafetyModel.psa

  RELAY_MALFUNCTION_ADDRS = {0: (LANE_KEEP_ASSIST,)}
  FWD_BLACKLISTED_ADDRS = {2: [LANE_KEEP_ASSIST]}
  TX_MSGS = [[1010, 0]]

  MAIN_BUS = 0
  ADAS_BUS = 1
  CAM_BUS = 2

  STEER_ANGLE_MAX = 390
  DEG_TO_CAN = 10

  ANGLE_RATE_BP = [0., 5., 25.]
  ANGLE_RATE_UP = [2.5, 1.5, .2]
  ANGLE_RATE_DOWN = [5., 2., .3]

  def _angle_cmd_msg(self, angle: float, enabled: bool):
    values = {"SET_ANGLE": angle, "TORQUE_FACTOR": 100 if enabled else 0}
    return self.packer.make_can_msg_safety("LANE_KEEP_ASSIST", self.MAIN_BUS, values)

  def _angle_meas_msg(self, angle: float):
    values = {"ANGLE": angle}
    return self.packer.make_can_msg_safety("STEERING_ALT", self.MAIN_BUS, values)

  def _pcm_status_msg(self, enable):
    values = {"RVV_ACC_ACTIVATION_REQ": enable}
    return self.packer.make_can_msg_safety("HS2_DAT_MDD_CMD_452", self.ADAS_BUS, values)

  def _speed_msg(self, speed):
    values = {"VITESSE_VEHICULE_ROUES": speed * 3.6}
    return self.packer.make_can_msg_safety("HS2_DYN_ABR_38D", self.MAIN_BUS, values)

  def _user_brake_msg(self, brake):
    values = {"P013_MainBrake": brake}
    return self.packer.make_can_msg_safety("Dat_BSI", self.CAM_BUS, values)

  def _user_gas_msg(self, gas):
    values = {"P002_Com_rAPP": int(gas * 100)}
    return self.packer.make_can_msg_safety("Dyn_CMM", self.MAIN_BUS, values)


class TestPsaIgnition(common.CanIgnitionSafetyTest):
  IGNITION_VALUES = range(4)
  IGNITION_ON_VALUES = {1, 2, 3}  # IS_DAT_BSI_432->ETAT_PRINCIP_SEV 0=stop, 1=ignition on, 2=cranking
  IGNITION_COUNTER_MOD = 16
  IGNITION_BUSES = (0, 2)  # the BSI is on bus 2 and only reaches bus 0 while the relay is open

  def _ignition_msg(self, value, counter=None, bus=0):
    assert counter is not None  # IGNITION_COUNTER_MOD is set
    return common.make_msg(bus, 0x432, dat=bytes([0, counter, 0, 0, 0, 0, value, 0]))


class TestPsaStockSafety(TestPsaSafetyBase):
  pass


if __name__ == "__main__":
    unittest.main()
