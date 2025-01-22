#!/usr/bin/env python3
import enum
import unittest
from panda import Panda
from panda.tests.libpanda import libpanda_py
import panda.tests.safety.common as common
from panda.tests.safety.common import CANPackerPanda
from functools import partial

class SubaruMsg(enum.IntEnum):
  Brake_Status      = 0x13c
  CruiseControl     = 0x240
  Throttle          = 0x40
  Steering_Torque   = 0x119
  Wheel_Speeds      = 0x13a
  ES_LKAS           = 0x122
  ES_LKAS_ANGLE     = 0x124
  ES_Brake          = 0x220
  ES_Distance       = 0x221
  ES_Status         = 0x222
  ES_DashStatus     = 0x321
  ES_LKAS_State     = 0x322
  ES_Infotainment   = 0x323
  ES_UDS_Request    = 0x787
  ES_HighBeamAssist = 0x121
  ES_STATIC_1       = 0x22a
  ES_STATIC_2       = 0x325


SUBARU_MAIN_BUS = 0
SUBARU_ALT_BUS  = 1
SUBARU_CAM_BUS  = 2


def lkas_tx_msgs(alt_bus, lkas_msg=SubaruMsg.ES_LKAS):
  return [[lkas_msg,                    SUBARU_MAIN_BUS],
          [SubaruMsg.ES_Distance,       alt_bus],
          [SubaruMsg.ES_DashStatus,     SUBARU_MAIN_BUS],
          [SubaruMsg.ES_LKAS_State,     SUBARU_MAIN_BUS],
          [SubaruMsg.ES_Infotainment,   SUBARU_MAIN_BUS]]

def long_tx_msgs(alt_bus):
  return [[SubaruMsg.ES_Brake,          alt_bus],
          [SubaruMsg.ES_Status,         alt_bus]]

def gen2_long_additional_tx_msgs():
  return [[SubaruMsg.ES_UDS_Request,    SUBARU_CAM_BUS],
          [SubaruMsg.ES_HighBeamAssist, SUBARU_MAIN_BUS],
          [SubaruMsg.ES_STATIC_1,       SUBARU_MAIN_BUS],
          [SubaruMsg.ES_STATIC_2,       SUBARU_MAIN_BUS]]

def fwd_blacklisted_addr(lkas_msg=SubaruMsg.ES_LKAS):
  return {SUBARU_CAM_BUS: [lkas_msg, SubaruMsg.ES_DashStatus, SubaruMsg.ES_LKAS_State, SubaruMsg.ES_Infotainment]}

class TestSubaruSafetyBase(common.PandaCarSafetyTest):
  FLAGS = 0
  STANDSTILL_THRESHOLD = 0 # kph
  RELAY_MALFUNCTION_ADDRS = {SUBARU_MAIN_BUS: (SubaruMsg.ES_LKAS,)}
  FWD_BUS_LOOKUP = {SUBARU_MAIN_BUS: SUBARU_CAM_BUS, SUBARU_CAM_BUS: SUBARU_MAIN_BUS}
  FWD_BLACKLISTED_ADDRS = fwd_blacklisted_addr()

  MAX_RT_DELTA = 940
  RT_INTERVAL = 250000

  DRIVER_TORQUE_ALLOWANCE = 60
  DRIVER_TORQUE_FACTOR = 50

  ALT_MAIN_BUS = SUBARU_MAIN_BUS
  ALT_CAM_BUS = SUBARU_CAM_BUS

  DEG_TO_CAN = 100

  INACTIVE_GAS = 1818

  def setUp(self):
    self.packer = CANPackerPanda("subaru_global_2017_generated")
    self.safety = libpanda_py.libpanda
    self.safety.set_safety_hooks(Panda.SAFETY_SUBARU, self.FLAGS)
    self.safety.init_tests()

  def _set_prev_torque(self, t):
    self.safety.set_desired_torque_last(t)
    self.safety.set_rt_torque_last(t)

  def _torque_driver_msg(self, torque):
    values = {"Steer_Torque_Sensor": torque}
    return self.packer.make_can_msg_panda("Steering_Torque", 0, values)

  def _speed_msg(self, speed):
    values = {s: speed for s in ["FR", "FL", "RR", "RL"]}
    return self.packer.make_can_msg_panda("Wheel_Speeds", self.ALT_MAIN_BUS, values)

  def _angle_meas_msg(self, angle):
    values = {"Steering_Angle": angle}
    return self.packer.make_can_msg_panda("Steering_Torque", 0, values)

  def _user_brake_msg(self, brake):
    values = {"Brake": brake}
    return self.packer.make_can_msg_panda("Brake_Status", self.ALT_MAIN_BUS, values)

  def _user_gas_msg(self, gas):
    values = {"Throttle_Pedal": gas}
    return self.packer.make_can_msg_panda("Throttle", 0, values)

  def _pcm_status_msg(self, enable):
    values = {"Cruise_Activated": enable}
    return self.packer.make_can_msg_panda("CruiseControl", self.ALT_MAIN_BUS, values)


class TestSubaruStockLongitudinalSafetyBase(TestSubaruSafetyBase):
  def _cancel_msg(self, cancel, cruise_throttle=0):
    values = {"Cruise_Cancel": cancel, "Cruise_Throttle": cruise_throttle}
    return self.packer.make_can_msg_panda("ES_Distance", self.ALT_MAIN_BUS, values)

  def test_cancel_message(self):
    # test that we can only send the cancel message (ES_Distance) with inactive throttle (1818) and Cruise_Cancel=1
    for cancel in [True, False]:
      self._generic_limit_safety_check(partial(self._cancel_msg, cancel), self.INACTIVE_GAS, self.INACTIVE_GAS, 0, 2**12, 1, self.INACTIVE_GAS, cancel)


class TestSubaruLongitudinalSafetyBase(TestSubaruSafetyBase, common.LongitudinalGasBrakeSafetyTest):
  MIN_GAS = 808
  MAX_GAS = 3400
  INACTIVE_GAS = 1818
  MAX_POSSIBLE_GAS = 2**13

  MIN_BRAKE = 0
  MAX_BRAKE = 600
  MAX_POSSIBLE_BRAKE = 2**16

  MIN_RPM = 0
  MAX_RPM = 3600
  MAX_POSSIBLE_RPM = 2**13

  FWD_BLACKLISTED_ADDRS = {2: [SubaruMsg.ES_LKAS, SubaruMsg.ES_Brake, SubaruMsg.ES_Distance,
                               SubaruMsg.ES_Status, SubaruMsg.ES_DashStatus,
                               SubaruMsg.ES_LKAS_State, SubaruMsg.ES_Infotainment]}

  def test_rpm_safety_check(self):
    self._generic_limit_safety_check(self._send_rpm_msg, self.MIN_RPM, self.MAX_RPM, 0, self.MAX_POSSIBLE_RPM, 1)

  def _send_brake_msg(self, brake):
    values = {"Brake_Pressure": brake}
    return self.packer.make_can_msg_panda("ES_Brake", self.ALT_MAIN_BUS, values)

  def _send_gas_msg(self, gas):
    values = {"Cruise_Throttle": gas}
    return self.packer.make_can_msg_panda("ES_Distance", self.ALT_MAIN_BUS, values)

  def _send_rpm_msg(self, rpm):
    values = {"Cruise_RPM": rpm}
    return self.packer.make_can_msg_panda("ES_Status", self.ALT_MAIN_BUS, values)


class TestSubaruTorqueSafetyBase(TestSubaruSafetyBase, common.DriverTorqueSteeringSafetyTest, common.SteerRequestCutSafetyTest):
  MAX_RATE_UP = 50
  MAX_RATE_DOWN = 70
  MAX_TORQUE = 2047

  # Safety around steering req bit
  MIN_VALID_STEERING_FRAMES = 7
  MAX_INVALID_STEERING_FRAMES = 1
  MIN_VALID_STEERING_RT_INTERVAL = 144000

  def _torque_cmd_msg(self, torque, steer_req=1):
    values = {"LKAS_Output": torque, "LKAS_Request": steer_req}
    return self.packer.make_can_msg_panda("ES_LKAS", SUBARU_MAIN_BUS, values)


class TestSubaruGen1TorqueStockLongitudinalSafety(TestSubaruStockLongitudinalSafetyBase, TestSubaruTorqueSafetyBase):
  FLAGS = 0
  TX_MSGS = lkas_tx_msgs(SUBARU_MAIN_BUS)


class TestSubaruGen2TorqueSafetyBase(TestSubaruTorqueSafetyBase):
  ALT_MAIN_BUS = SUBARU_ALT_BUS
  ALT_CAM_BUS = SUBARU_ALT_BUS

  MAX_RATE_UP = 40
  MAX_RATE_DOWN = 40
  MAX_TORQUE = 1000


class TestSubaruGen2TorqueStockLongitudinalSafety(TestSubaruStockLongitudinalSafetyBase, TestSubaruGen2TorqueSafetyBase):
  FLAGS = Panda.FLAG_SUBARU_GEN2
  TX_MSGS = lkas_tx_msgs(SUBARU_ALT_BUS)


class TestSubaruGen1LongitudinalSafety(TestSubaruLongitudinalSafetyBase, TestSubaruTorqueSafetyBase):
  FLAGS = Panda.FLAG_SUBARU_LONG
  TX_MSGS = lkas_tx_msgs(SUBARU_MAIN_BUS) + long_tx_msgs(SUBARU_MAIN_BUS)


class TestSubaruGen2LongitudinalSafety(TestSubaruLongitudinalSafetyBase, TestSubaruGen2TorqueSafetyBase):
  FLAGS = Panda.FLAG_SUBARU_LONG | Panda.FLAG_SUBARU_GEN2
  TX_MSGS = lkas_tx_msgs(SUBARU_ALT_BUS) + long_tx_msgs(SUBARU_ALT_BUS) + gen2_long_additional_tx_msgs()

  def _rdbi_msg(self, did: int):
    return b'\x03\x22' + did.to_bytes(2) + b'\x00\x00\x00\x00'

  def _es_uds_msg(self, msg: bytes):
    return libpanda_py.make_CANPacket(SubaruMsg.ES_UDS_Request, 2, msg)

  def test_es_uds_message(self):
    tester_present = b'\x02\x3E\x80\x00\x00\x00\x00\x00'
    not_tester_present = b"\x03\xAA\xAA\x00\x00\x00\x00\x00"

    button_did = 0x1130

    # Tester present is allowed for gen2 long to keep eyesight disabled
    self.assertTrue(self._tx(self._es_uds_msg(tester_present)))

    # Non-Tester present is not allowed
    self.assertFalse(self._tx(self._es_uds_msg(not_tester_present)))

    # Only button_did is allowed to be read via UDS
    for did in range(0xFFFF):
      should_tx = (did == button_did)
      self.assertEqual(self._tx(self._es_uds_msg(self._rdbi_msg(did))), should_tx)

    # any other msg is not allowed
    for sid in range(0xFF):
      msg = b'\x03' + sid.to_bytes(1) + b'\x00' * 6
      self.assertFalse(self._tx(self._es_uds_msg(msg)))


if __name__ == "__main__":
  unittest.main()
