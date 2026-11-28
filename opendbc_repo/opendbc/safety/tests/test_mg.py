#!/usr/bin/env python3
import unittest

from opendbc.car.structs import CarParams
import opendbc.safety.tests.common as common
from opendbc.safety.tests.libsafety import libsafety_py


class TestMGSafety(common.CarSafetyTest, common.DriverTorqueSteeringSafetyTest):

  DBC = "mg"
  SAFETY_MODEL = CarParams.SafetyModel.mg

  TX_MSGS = [[0x1fd, 0], ]
  RELAY_MALFUNCTION_ADDRS = {0: (0x1fd,)}
  FWD_BLACKLISTED_ADDRS = {2: [0x1fd,]}

  MAX_RATE_UP = 6
  MAX_RATE_DOWN = 10
  MAX_TORQUE_LOOKUP = [0], [300]
  MAX_RT_DELTA = 125

  DRIVER_TORQUE_ALLOWANCE = 100
  DRIVER_TORQUE_FACTOR = 2

  def _torque_cmd_msg(self, torque, steer_req=1):
    values = {"LKAReqToqHSC2": torque, "LKAReqToqStsHSC2": steer_req}
    return self.packer.make_can_msg_safety("FVCM_HSC2_FrP03", 0, values)

  def _speed_msg(self, speed):
    values = {"VehSpdAvgHSC2": speed * 3.6}
    return self.packer.make_can_msg_safety("SCS_HSC2_FrP19", 0, values)

  def _torque_driver_msg(self, torque):
    values = {"DrvrStrgDlvrdToqHSC2": torque * 0.01}
    return self.packer.make_can_msg_safety("EPS_HSC2_FrP03", 0, values)

  def _user_brake_msg(self, brake):
    values = {"BrkPdlAppdHSC2": 1 if brake else 0}
    return self.packer.make_can_msg_safety("EHBS_HSC2_FrP00", 0, values)

  def _user_gas_msg(self, gas):
    values = {"EPTAccelActuPosHSC2": 100 if gas else 0}
    return self.packer.make_can_msg_safety("GW_HSC2_HCU_FrP00", 0, values)

  def _pcm_status_msg(self, enable):
    values = {"ACCSysSts_RadarHSC2": 2 if enable else 1}
    return self.packer.make_can_msg_safety("RADAR_HSC2_FrP00", 0, values)

  def test_cruise_states(self):
    for state in range(8):
      self._rx(self._pcm_status_msg(False))
      values = {"ACCSysSts_RadarHSC2": state}
      self._rx(self.packer.make_can_msg_safety("RADAR_HSC2_FrP00", 0, values))
      self.assertEqual(state in (2, 3), self.safety.get_controls_allowed())

  def test_eps_checksum_captured_frame(self):
    # a captured EPS frame: the packer computes its PV field 0x37f5, and the safety accepts it
    dat = bytes.fromhex("b40037f553f54000")
    values = {"ChLKAAlvRCHSC2": 11, "ChLKACtrlStsHSC2": 4, "ChLKARespToqHSC2": 0}
    packed = self.packer.make_can_msg_safety("EPS_HSC2_FrP03", 0, values)
    self.assertEqual(bytes(packed[0].data)[2:4], dat[2:4])
    self.assertTrue(self._rx(libsafety_py.make_CANPacket(0x1ec, 0, dat)))


if __name__ == "__main__":
  unittest.main()
