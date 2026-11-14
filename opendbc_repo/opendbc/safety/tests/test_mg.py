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

  def setUp(self):
    super().setUp()
    self.counters = {addr: 0 for addr in (0x1b6, 0x1ec, 0x23c, 0x242)}
    self.gas_counter = 0

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
    return self.packer.make_can_msg_safety("EHBS_HSC2_FrP00", 0, values)

  def _user_gas_msg(self, gas):
    values = {"EPTAccelActuPosHSC2": 100 if gas else 0}
    msg = self.packer.make_can_msg_safety("GW_HSC2_HCU_FrP00", 0, values)
    msg[0].data[5] = (msg[0].data[5] & 0x0F) | ((self.gas_counter % 16) << 4)
    self.gas_counter += 1
    return msg

  def _pcm_status_msg(self, enable):
    values = {"ACCSysSts_RadarHSC2": 2 if enable else 1, "ACCSysAlvRlngCtr_SCSHSC2": self._counter(0x242)}
    return self.packer.make_can_msg_safety("RADAR_HSC2_FrP00", 0, values)

  def test_gas_counter(self):
    self._reset_safety_hooks()
    for _ in range(16):
      self.assertTrue(self._rx(self._user_gas_msg(0)))

    msg = self._user_gas_msg(0)
    for _ in range(common.MAX_WRONG_COUNTERS + 1):
      valid = self._rx(msg)
    self.assertFalse(valid)

  def test_rx_checksums(self):
    # Captured EPS frame: the PV field is 0x37f5; byte 7 is unused.
    dat = bytes.fromhex("b40037f553f54000")
    values = {"ChLKAAlvRCHSC2": 11, "ChLKACtrlStsHSC2": 4, "ChLKARespToqHSC2": 0}
    packed = self.packer.make_can_msg_safety("EPS_HSC2_FrP03", 0, values)
    self.assertEqual(bytes(packed[0].data)[2:4], dat[2:4])

    for byte, bit in ((0, 0), (0, 3), (0, 4), (1, 0), (2, 0), (3, 0), (6, 4)):
      self._reset_safety_hooks()
      self.assertTrue(self._rx(libsafety_py.make_CANPacket(0x1ec, 0, dat)))
      corrupt = bytearray(dat)
      corrupt[byte] ^= 1 << bit
      self.assertFalse(self._rx(libsafety_py.make_CANPacket(0x1ec, 0, corrupt)))

    for make_msg in (self._speed_msg, self._torque_driver_msg, self._user_brake_msg, self._pcm_status_msg):
      self._reset_safety_hooks()
      self.assertTrue(self._rx(make_msg(0)))

      msg = make_msg(0)
      msg[0].data[3 if make_msg == self._torque_driver_msg else 7] ^= 0xff
      self.assertFalse(self._rx(msg))


if __name__ == "__main__":
  unittest.main()
