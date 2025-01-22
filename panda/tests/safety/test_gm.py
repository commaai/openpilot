#!/usr/bin/env python3
import unittest
from panda import Panda
from panda.tests.libpanda import libpanda_py
import panda.tests.safety.common as common
from panda.tests.safety.common import CANPackerPanda


class Buttons:
  UNPRESS = 1
  RES_ACCEL = 2
  DECEL_SET = 3
  CANCEL = 6


class GmLongitudinalBase(common.PandaCarSafetyTest, common.LongitudinalGasBrakeSafetyTest):
  # pylint: disable=no-member,abstract-method

  RELAY_MALFUNCTION_ADDRS = {0: (0x180, 0x2CB)}  # ASCMLKASteeringCmd, ASCMGasRegenCmd

  MAX_POSSIBLE_BRAKE = 2 ** 12
  MAX_BRAKE = 400

  MAX_POSSIBLE_GAS = 2 ** 12

  PCM_CRUISE = False  # openpilot can control the PCM state if longitudinal

  def _send_brake_msg(self, brake):
    values = {"FrictionBrakeCmd": -brake}
    return self.packer_chassis.make_can_msg_panda("EBCMFrictionBrakeCmd", self.BRAKE_BUS, values)

  def _send_gas_msg(self, gas):
    values = {"GasRegenCmd": gas}
    return self.packer.make_can_msg_panda("ASCMGasRegenCmd", 0, values)

  # override these tests from PandaCarSafetyTest, GM longitudinal uses button enable
  def _pcm_status_msg(self, enable):
    raise NotImplementedError

  def test_disable_control_allowed_from_cruise(self):
    pass

  def test_enable_control_allowed_from_cruise(self):
    pass

  def test_cruise_engaged_prev(self):
    pass

  def test_set_resume_buttons(self):
    """
      SET and RESUME enter controls allowed on their falling and rising edges, respectively.
    """
    for btn_prev in range(8):
      for btn_cur in range(8):
        with self.subTest(btn_prev=btn_prev, btn_cur=btn_cur):
          self._rx(self._button_msg(btn_prev))
          self.safety.set_controls_allowed(0)
          for _ in range(10):
            self._rx(self._button_msg(btn_cur))

          should_enable = btn_cur != Buttons.DECEL_SET and btn_prev == Buttons.DECEL_SET
          should_enable = should_enable or (btn_cur == Buttons.RES_ACCEL and btn_prev != Buttons.RES_ACCEL)
          should_enable = should_enable and btn_cur != Buttons.CANCEL
          self.assertEqual(should_enable, self.safety.get_controls_allowed())

  def test_cancel_button(self):
    self.safety.set_controls_allowed(1)
    self._rx(self._button_msg(Buttons.CANCEL))
    self.assertFalse(self.safety.get_controls_allowed())


class TestGmSafetyBase(common.PandaCarSafetyTest, common.DriverTorqueSteeringSafetyTest):
  STANDSTILL_THRESHOLD = 10 * 0.0311
  # Ensures ASCM is off on ASCM cars, and relay is not malfunctioning for camera-ACC cars
  RELAY_MALFUNCTION_ADDRS = {0: (0x180,)}  # ASCMLKASteeringCmd
  BUTTONS_BUS = 0  # rx or tx
  BRAKE_BUS = 0  # tx only

  MAX_RATE_UP = 10
  MAX_RATE_DOWN = 15
  MAX_TORQUE = 300
  MAX_RT_DELTA = 128
  RT_INTERVAL = 250000
  DRIVER_TORQUE_ALLOWANCE = 65
  DRIVER_TORQUE_FACTOR = 4

  PCM_CRUISE = True  # openpilot is tied to the PCM state if not longitudinal

  @classmethod
  def setUpClass(cls):
    if cls.__name__ == "TestGmSafetyBase":
      cls.packer = None
      cls.safety = None
      raise unittest.SkipTest

  def setUp(self):
    self.packer = CANPackerPanda("gm_global_a_powertrain_generated")
    self.packer_chassis = CANPackerPanda("gm_global_a_chassis")
    self.safety = libpanda_py.libpanda
    self.safety.set_safety_hooks(Panda.SAFETY_GM, 0)
    self.safety.init_tests()

  def _pcm_status_msg(self, enable):
    if self.PCM_CRUISE:
      values = {"CruiseState": enable}
      return self.packer.make_can_msg_panda("AcceleratorPedal2", 0, values)
    else:
      raise NotImplementedError

  def _speed_msg(self, speed):
    values = {"%sWheelSpd" % s: speed for s in ["RL", "RR"]}
    return self.packer.make_can_msg_panda("EBCMWheelSpdRear", 0, values)

  def _user_brake_msg(self, brake):
    # GM safety has a brake threshold of 8
    values = {"BrakePedalPos": 8 if brake else 0}
    return self.packer.make_can_msg_panda("ECMAcceleratorPos", 0, values)

  def _user_regen_msg(self, regen):
    values = {"RegenPaddle": 2 if regen else 0}
    return self.packer.make_can_msg_panda("EBCMRegenPaddle", 0, values)

  def _user_gas_msg(self, gas):
    values = {"AcceleratorPedal2": 1 if gas else 0}
    if self.PCM_CRUISE:
      # Fill CruiseState with expected value if the safety mode reads cruise state from gas msg
      values["CruiseState"] = self.safety.get_controls_allowed()
    return self.packer.make_can_msg_panda("AcceleratorPedal2", 0, values)

  def _torque_driver_msg(self, torque):
    # Safety tests assume driver torque is an int, use DBC factor
    values = {"LKADriverAppldTrq": torque * 0.01}
    return self.packer.make_can_msg_panda("PSCMStatus", 0, values)

  def _torque_cmd_msg(self, torque, steer_req=1):
    values = {"LKASteeringCmd": torque, "LKASteeringCmdActive": steer_req}
    return self.packer.make_can_msg_panda("ASCMLKASteeringCmd", 0, values)

  def _button_msg(self, buttons):
    values = {"ACCButtons": buttons}
    return self.packer.make_can_msg_panda("ASCMSteeringButton", self.BUTTONS_BUS, values)


class TestGmAscmSafety(GmLongitudinalBase, TestGmSafetyBase):
  TX_MSGS = [[0x180, 0], [0x409, 0], [0x40A, 0], [0x2CB, 0], [0x370, 0],  # pt bus
             [0xA1, 1], [0x306, 1], [0x308, 1], [0x310, 1],  # obs bus
             [0x315, 2]]  # ch bus
  FWD_BLACKLISTED_ADDRS: dict[int, list[int]] = {}
  FWD_BUS_LOOKUP: dict[int, int] = {}
  BRAKE_BUS = 2

  MAX_GAS = 3072
  MIN_GAS = 1404 # maximum regen
  INACTIVE_GAS = 1404

  def setUp(self):
    self.packer = CANPackerPanda("gm_global_a_powertrain_generated")
    self.packer_chassis = CANPackerPanda("gm_global_a_chassis")
    self.safety = libpanda_py.libpanda
    self.safety.set_safety_hooks(Panda.SAFETY_GM, 0)
    self.safety.init_tests()


class TestGmCameraSafetyBase(TestGmSafetyBase):

  FWD_BUS_LOOKUP = {0: 2, 2: 0}

  @classmethod
  def setUpClass(cls):
    if cls.__name__ == "TestGmCameraSafetyBase":
      cls.packer = None
      cls.safety = None
      raise unittest.SkipTest

  def _user_brake_msg(self, brake):
    values = {"BrakePressed": brake}
    return self.packer.make_can_msg_panda("ECMEngineStatus", 0, values)


class TestGmCameraSafety(TestGmCameraSafetyBase):
  TX_MSGS = [[0x180, 0],  # pt bus
             [0x184, 2]]  # camera bus
  FWD_BLACKLISTED_ADDRS = {2: [0x180], 0: [0x184]}  # block LKAS message and PSCMStatus
  BUTTONS_BUS = 2  # tx only

  def setUp(self):
    self.packer = CANPackerPanda("gm_global_a_powertrain_generated")
    self.packer_chassis = CANPackerPanda("gm_global_a_chassis")
    self.safety = libpanda_py.libpanda
    self.safety.set_safety_hooks(Panda.SAFETY_GM, Panda.FLAG_GM_HW_CAM)
    self.safety.init_tests()

  def test_buttons(self):
    # Only CANCEL button is allowed while cruise is enabled
    self.safety.set_controls_allowed(0)
    for btn in range(8):
      self.assertFalse(self._tx(self._button_msg(btn)))

    self.safety.set_controls_allowed(1)
    for btn in range(8):
      self.assertFalse(self._tx(self._button_msg(btn)))

    for enabled in (True, False):
      self._rx(self._pcm_status_msg(enabled))
      self.assertEqual(enabled, self._tx(self._button_msg(Buttons.CANCEL)))


class TestGmCameraLongitudinalSafety(GmLongitudinalBase, TestGmCameraSafetyBase):
  TX_MSGS = [[0x180, 0], [0x315, 0], [0x2CB, 0], [0x370, 0],  # pt bus
             [0x184, 2]]  # camera bus
  FWD_BLACKLISTED_ADDRS = {2: [0x180, 0x2CB, 0x370, 0x315], 0: [0x184]}  # block LKAS, ACC messages and PSCMStatus
  BUTTONS_BUS = 0  # rx only

  MAX_GAS = 3400
  MIN_GAS = 1514 # maximum regen
  INACTIVE_GAS = 1554

  def setUp(self):
    self.packer = CANPackerPanda("gm_global_a_powertrain_generated")
    self.packer_chassis = CANPackerPanda("gm_global_a_chassis")
    self.safety = libpanda_py.libpanda
    self.safety.set_safety_hooks(Panda.SAFETY_GM, Panda.FLAG_GM_HW_CAM | Panda.FLAG_GM_HW_CAM_LONG)
    self.safety.init_tests()


if __name__ == "__main__":
  unittest.main()
