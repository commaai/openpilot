#!/usr/bin/env python3
import random
import unittest

from opendbc.car.hyundai.values import HyundaiSafetyFlags
from opendbc.car.structs import CarParams
import opendbc.safety.tests.common as common
import opendbc.safety.tests.hyundai_common as hyundai_common


class TestHyundaiSafety(hyundai_common.HyundaiButtonBase, common.CarSafetyTest, common.DriverTorqueSteeringSafetyTest, common.SteerRequestCutSafetyTest):
  DBC = "hyundai_can_generated"
  SAFETY_MODEL = CarParams.SafetyModel.hyundai

  TX_MSGS = [[0x340, 0], [0x4F1, 0], [0x485, 0]]
  STANDSTILL_THRESHOLD = 12  # 0.375 kph
  RELAY_MALFUNCTION_ADDRS = {0: (0x340, 0x485)}  # LKAS11
  FWD_BLACKLISTED_ADDRS = {2: [0x340, 0x485]}

  MAX_RATE_UP = 3
  MAX_RATE_DOWN = 7
  MAX_TORQUE_LOOKUP = [0], [384]
  MAX_RT_DELTA = 112
  DRIVER_TORQUE_ALLOWANCE = 50
  DRIVER_TORQUE_FACTOR = 2

  # Safety around steering req bit
  MIN_VALID_STEERING_FRAMES = 89
  MAX_INVALID_STEERING_FRAMES = 2

  def _button_msg(self, buttons, main_button=0, bus=0):
    values = {"CF_Clu_CruiseSwState": buttons, "CF_Clu_CruiseSwMain": main_button}
    return self.packer.make_can_msg_safety("CLU11", bus, values)

  def _user_gas_msg(self, gas):
    values = {"CF_Ems_AclAct": gas}
    return self.packer.make_can_msg_safety("EMS16", 0, values)

  def _user_brake_msg(self, brake):
    values = {"DriverOverride": 2 if brake else random.choice((0, 1, 3))}
    return self.packer.make_can_msg_safety("TCS13", 0, values)

  def _speed_msg(self, speed):
    # safety doesn't scale, so undo the scaling. It only reads two wheels
    values = {"WHL_SPD_%s" % s: speed * 0.03125 for s in ("FL", "RR")}
    return self.packer.make_can_msg_safety("WHL_SPD11", 0, values)

  def _pcm_status_msg(self, enable):
    values = {"ACCMode": enable}
    return self.packer.make_can_msg_safety("SCC12", self.SCC_BUS, values)

  def _torque_driver_msg(self, torque):
    values = {"CR_Mdps_StrColTq": torque}
    return self.packer.make_can_msg_safety("MDPS12", 0, values)

  def _torque_cmd_msg(self, torque, steer_req=1):
    values = {"CR_Lkas_StrToqReq": torque, "CF_Lkas_ActToi": steer_req}
    return self.packer.make_can_msg_safety("LKAS11", 0, values)

  def test_other_gas_msgs_ignored(self):
    # E_EMS11 shares the gas RX slot, but is only parsed with the EV or hybrid gas flags
    if self.SAFETY_PARAM & (HyundaiSafetyFlags.EV_GAS | HyundaiSafetyFlags.HYBRID_GAS):
      raise unittest.SkipTest
    self.assertTrue(self._rx(self.packer.make_can_msg_safety("E_EMS11", 0, {"Accel_Pedal_Pos": 1, "CR_Vcu_AccPedDep_Pos": 1})))
    self.assertFalse(self.safety.get_gas_pressed_prev())


class TestHyundaiSafetyAltLimits(TestHyundaiSafety):
  SAFETY_PARAM = HyundaiSafetyFlags.ALT_LIMITS

  MAX_RATE_UP = 2
  MAX_RATE_DOWN = 3
  MAX_TORQUE_LOOKUP = [0], [270]

class TestHyundaiSafetyAltLimits2(TestHyundaiSafety):
  SAFETY_PARAM = HyundaiSafetyFlags.ALT_LIMITS_2

  MAX_RATE_UP = 2
  MAX_RATE_DOWN = 3
  MAX_TORQUE_LOOKUP = [0], [170]

class TestHyundaiSafetyCameraSCC(TestHyundaiSafety):
  SAFETY_PARAM = HyundaiSafetyFlags.CAMERA_SCC

  BUTTONS_TX_BUS = 2  # tx on 2, rx on 0
  SCC_BUS = 2  # rx on 2

class TestHyundaiSafetyFCEV(TestHyundaiSafety):
  SAFETY_PARAM = HyundaiSafetyFlags.FCEV_GAS

  def _user_gas_msg(self, gas):
    values = {"ACCELERATOR_PEDAL": gas}
    return self.packer.make_can_msg_safety("FCEV_ACCELERATOR", 0, values)


class TestHyundaiLegacySafety(TestHyundaiSafety):
  SAFETY_MODEL = CarParams.SafetyModel.hyundaiLegacy

class TestHyundaiLegacySafetyEV(TestHyundaiSafety):
  SAFETY_MODEL = CarParams.SafetyModel.hyundaiLegacy
  SAFETY_PARAM = HyundaiSafetyFlags.EV_GAS

  def _user_gas_msg(self, gas):
    values = {"Accel_Pedal_Pos": gas}
    return self.packer.make_can_msg_safety("E_EMS11", 0, values)


class TestHyundaiLegacySafetyHEV(TestHyundaiSafety):
  SAFETY_MODEL = CarParams.SafetyModel.hyundaiLegacy
  SAFETY_PARAM = HyundaiSafetyFlags.HYBRID_GAS

  def _user_gas_msg(self, gas):
    values = {"CR_Vcu_AccPedDep_Pos": gas}
    return self.packer.make_can_msg_safety("E_EMS11", 0, values)


class TestHyundaiLongitudinalSafety(hyundai_common.HyundaiLongitudinalBase, TestHyundaiSafety):
  SAFETY_PARAM = HyundaiSafetyFlags.LONG

  TX_MSGS = [[0x340, 0], [0x4F1, 0], [0x485, 0], [0x420, 0], [0x421, 0], [0x50A, 0], [0x389, 0], [0x4A2, 0], [0x38D, 0], [0x483, 0], [0x7D0, 0]]

  FWD_BLACKLISTED_ADDRS = {2: [0x340, 0x485, 0x421, 0x420, 0x50A, 0x389]}

  RELAY_MALFUNCTION_ADDRS = {0: (0x340, 0x485, 0x421, 0x420, 0x50A, 0x389)}  # LKAS11, LFAHDA_MFC, SCC12, SCC11, SCC13, SCC14

  DISABLED_ECU_UDS_MSG = (0x7D0, 0)

  def test_button_sends(self):
    # buttons aren't restricted with openpilot longitudinal
    for btn in range(8):
      self.assertTrue(self._tx(self._button_msg(btn)))

  def _accel_msg(self, accel, aeb_req=False, aeb_decel=0, aeb_stop_req=False, accel_value=None):
    values = {
      "aReqRaw": accel,
      "aReqValue": accel if accel_value is None else accel_value,
      "AEB_CmdAct": int(aeb_req),
      "AEB_StopReq": int(aeb_stop_req),
      "CR_VSM_DecCmd": aeb_decel,
    }
    return self.packer.make_can_msg_safety("SCC12", self.SCC_BUS, values)

  def _fca11_msg(self, idx=0, vsm_aeb_req=False, fca_aeb_req=False, aeb_decel=0):
    values = {
      "CR_FCA_Alive": idx % 0xF,
      "FCA_Status": 2,
      "CR_VSM_DecCmd": aeb_decel,
      "CF_VSM_DecCmdAct": int(vsm_aeb_req),
      "FCA_CmdAct": int(fca_aeb_req),
    }
    return self.packer.make_can_msg_safety("FCA11", 0, values)

  def test_no_aeb_fca11(self):
    self.assertTrue(self._tx(self._fca11_msg()))
    self.assertFalse(self._tx(self._fca11_msg(vsm_aeb_req=True)))
    self.assertFalse(self._tx(self._fca11_msg(fca_aeb_req=True)))
    self.assertFalse(self._tx(self._fca11_msg(aeb_decel=1.0)))

  def test_no_aeb_scc12(self):
    self.assertTrue(self._tx(self._accel_msg(0)))
    self.assertFalse(self._tx(self._accel_msg(0, aeb_req=True)))
    self.assertFalse(self._tx(self._accel_msg(0, aeb_decel=1.0)))
    self.assertFalse(self._tx(self._accel_msg(0, aeb_stop_req=True)))

  def test_accel_signals_checked_independently(self):
    self.safety.set_controls_allowed(True)
    self.assertTrue(self._tx(self._accel_msg(0, accel_value=0)))
    self.assertFalse(self._tx(self._accel_msg(0, accel_value=5.0)))
    self.assertFalse(self._tx(self._accel_msg(5.0, accel_value=0)))


class TestHyundaiLongitudinalSafetyCameraSCC(hyundai_common.HyundaiLongitudinalBase, TestHyundaiSafety):
  SAFETY_PARAM = HyundaiSafetyFlags.LONG | HyundaiSafetyFlags.CAMERA_SCC

  TX_MSGS = [[0x340, 0], [0x4F1, 2], [0x485, 0], [0x420, 0], [0x421, 0], [0x50A, 0], [0x389, 0], [0x4A2, 0]]

  FWD_BLACKLISTED_ADDRS = {2: [0x340, 0x485, 0x420, 0x421, 0x50A, 0x389]}
  RELAY_MALFUNCTION_ADDRS = {0: (0x340, 0x485, 0x421, 0x420, 0x50A, 0x389)}  # LKAS11, LFAHDA_MFC, SCC12, SCC11, SCC13, SCC14

  def _accel_msg(self, accel, aeb_req=False, aeb_decel=0, aeb_stop_req=False, accel_value=None):
    values = {
      "aReqRaw": accel,
      "aReqValue": accel if accel_value is None else accel_value,
      "AEB_CmdAct": int(aeb_req),
      "AEB_StopReq": int(aeb_stop_req),
      "CR_VSM_DecCmd": aeb_decel,
    }
    return self.packer.make_can_msg_safety("SCC12", self.SCC_BUS, values)

  def test_no_aeb_scc12(self):
    self.assertTrue(self._tx(self._accel_msg(0)))
    self.assertFalse(self._tx(self._accel_msg(0, aeb_req=True)))
    self.assertFalse(self._tx(self._accel_msg(0, aeb_decel=1.0)))
    self.assertFalse(self._tx(self._accel_msg(0, aeb_stop_req=True)))

  def test_accel_signals_checked_independently(self):
    self.safety.set_controls_allowed(True)
    self.assertTrue(self._tx(self._accel_msg(0, accel_value=0)))
    self.assertFalse(self._tx(self._accel_msg(0, accel_value=5.0)))
    self.assertFalse(self._tx(self._accel_msg(5.0, accel_value=0)))

  def test_tester_present_allowed(self):
    pass


class TestHyundaiSafetyFCEVLong(TestHyundaiLongitudinalSafety, TestHyundaiSafetyFCEV):
  SAFETY_PARAM = HyundaiSafetyFlags.FCEV_GAS | HyundaiSafetyFlags.LONG


if __name__ == "__main__":
  unittest.main()
