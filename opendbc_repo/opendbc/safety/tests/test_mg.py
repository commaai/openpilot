#!/usr/bin/env python3
import unittest

from opendbc.car.structs import CarParams
from opendbc.safety.tests.libsafety import libsafety_py
import opendbc.safety.tests.common as common
from opendbc.safety.tests.common import CANPackerSafety


def checksum(msg):
  addr, dat, bus = msg
  ret = bytearray(dat)

  if addr in (0x1b6, 0x242):
    crc = 0xFF
    for byte in ret[:-1]:
      crc ^= byte
      for _ in range(8):
        crc = ((crc << 1) ^ 0x1D) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    ret[-1] = crc ^ 0xFF

  return addr, ret, bus


class TestMGSafety(common.CarSafetyTest, common.DriverTorqueSteeringSafetyTest):

  TX_MSGS = [[0x1fd, 0], ]
  RELAY_MALFUNCTION_ADDRS = {0: (0x1fd,)}
  FWD_BLACKLISTED_ADDRS = {2: [0x1fd,]}

  MAX_RATE_UP = 6
  MAX_RATE_DOWN = 10
  MAX_TORQUE_LOOKUP = [0], [300]
  MAX_RT_DELTA = 125

  DRIVER_TORQUE_ALLOWANCE = 100
  DRIVER_TORQUE_FACTOR = 2

  def setUp(self):
    self.packer = CANPackerSafety("mg")
    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(CarParams.SafetyModel.mg, 0)
    self.safety.init_tests()
    self.counters = {addr: 0 for addr in (0x1b6, 0x1ec, 0x23c, 0x242)}

  def _counter(self, addr):
    counter = self.counters[addr]
    self.counters[addr] = (counter + 1) % 16
    return counter

  def _torque_cmd_msg(self, torque, steer_req=1):
    values = {"LKAReqToqHSC2": torque, "LKAReqToqStsHSC2": steer_req}
    return self.packer.make_can_msg_safety("FVCM_HSC2_FrP03", 0, values)

  def _speed_msg(self, speed):
    values = {"VehSpdAvgHSC2": speed * 3.6, "VehSpdAvgAlvRCHSC2": self._counter(0x23c)}
    return self.packer.make_can_msg_safety("SCS_HSC2_FrP19", 0, values)

  def _torque_driver_msg(self, torque):
    values = {"DrvrStrgDlvrdToqHSC2": torque * 0.01, "ChLKAAlvRCHSC2": self._counter(0x1ec)}
    return self.packer.make_can_msg_safety("EPS_HSC2_FrP03", 0, values)

  def _user_brake_msg(self, brake):
    values = {"BrkPdlAppdHSC2": 1 if brake else 0, "BrkPdlAppdRCHSC2": self._counter(0x1b6)}
    return self.packer.make_can_msg_safety("EHBS_HSC2_FrP00", 0, values, fix_checksum=checksum)

  def _user_gas_msg(self, gas):
    values = {"EPTAccelActuPosHSC2": 100 if gas else 0}
    return self.packer.make_can_msg_safety("GW_HSC2_HCU_FrP00", 0, values)

  def _pcm_status_msg(self, enable):
    values = {"ACCSysSts_RadarHSC2": 2 if enable else 1, "ACCSysAlvRlngCtr_SCSHSC2": self._counter(0x242)}
    return self.packer.make_can_msg_safety("RADAR_HSC2_FrP00", 0, values, fix_checksum=checksum)


if __name__ == "__main__":
  unittest.main()
