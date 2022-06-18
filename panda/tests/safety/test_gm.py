#!/usr/bin/env python3
import unittest
from typing import Dict, List
from panda import Panda
from panda.tests.safety import libpandasafety_py
import panda.tests.safety.common as common
from panda.tests.safety.common import CANPackerPanda, ALTERNATIVE_EXPERIENCE

MAX_BRAKE = 350
MAX_GAS = 3072
MAX_REGEN = 1404


class Buttons:
  UNPRESS = 1
  RES_ACCEL = 2
  DECEL_SET = 3
  CANCEL = 6


class TestGmSafety(common.PandaSafetyTest, common.DriverTorqueSteeringSafetyTest):
  TX_MSGS = [[384, 0], [1033, 0], [1034, 0], [715, 0], [880, 0],  # pt bus
             [161, 1], [774, 1], [776, 1], [784, 1],  # obs bus
             [789, 2],  # ch bus
             [0x104c006c, 3], [0x10400060, 3]]  # gmlan
  STANDSTILL_THRESHOLD = 0
  RELAY_MALFUNCTION_ADDR = 384
  RELAY_MALFUNCTION_BUS = 0
  FWD_BLACKLISTED_ADDRS: Dict[int, List[int]] = {}
  FWD_BUS_LOOKUP: Dict[int, int] = {}

  MAX_RATE_UP = 7
  MAX_RATE_DOWN = 17
  MAX_TORQUE = 300
  MAX_RT_DELTA = 128
  RT_INTERVAL = 250000
  DRIVER_TORQUE_ALLOWANCE = 50
  DRIVER_TORQUE_FACTOR = 4

  def setUp(self):
    self.packer = CANPackerPanda("gm_global_a_powertrain_generated")
    self.packer_chassis = CANPackerPanda("gm_global_a_chassis")
    self.safety = libpandasafety_py.libpandasafety
    self.safety.set_safety_hooks(Panda.SAFETY_GM, 0)
    self.safety.init_tests()

  # override these tests from PandaSafetyTest, GM uses button enable
  def test_disable_control_allowed_from_cruise(self):
    pass

  def test_enable_control_allowed_from_cruise(self):
    pass

  def test_cruise_engaged_prev(self):
    pass

  def _pcm_status_msg(self, enable):
    raise NotImplementedError

  def _speed_msg(self, speed):
    values = {"%sWheelSpd" % s: speed for s in ["RL", "RR"]}
    return self.packer.make_can_msg_panda("EBCMWheelSpdRear", 0, values)

  def _button_msg(self, buttons):
    values = {"ACCButtons": buttons}
    return self.packer.make_can_msg_panda("ASCMSteeringButton", 0, values)

  def _user_brake_msg(self, brake):
    # GM safety has a brake threshold of 10
    values = {"BrakePedalPosition": 10 if brake else 0}
    return self.packer.make_can_msg_panda("EBCMBrakePedalPosition", 0, values)

  def _user_gas_msg(self, gas):
    values = {"AcceleratorPedal2": 1 if gas else 0}
    return self.packer.make_can_msg_panda("AcceleratorPedal2", 0, values)

  def _send_brake_msg(self, brake):
    values = {"FrictionBrakeCmd": -brake}
    return self.packer_chassis.make_can_msg_panda("EBCMFrictionBrakeCmd", 2, values)

  def _send_gas_msg(self, gas):
    values = {"GasRegenCmd": gas}
    return self.packer.make_can_msg_panda("ASCMGasRegenCmd", 0, values)

  def _torque_driver_msg(self, torque):
    values = {"LKADriverAppldTrq": torque}
    return self.packer.make_can_msg_panda("PSCMStatus", 0, values)

  def _torque_cmd_msg(self, torque, steer_req=1):
    values = {"LKASteeringCmd": torque}
    return self.packer.make_can_msg_panda("ASCMLKASteeringCmd", 0, values)

  def test_set_resume_buttons(self):
    """
      SET and RESUME enter controls allowed on their falling edge.
    """
    for btn in range(8):
      self.safety.set_controls_allowed(0)
      for _ in range(10):
        self._rx(self._button_msg(btn))
        self.assertFalse(self.safety.get_controls_allowed())

      # should enter controls allowed on falling edge
      if btn in (Buttons.RES_ACCEL, Buttons.DECEL_SET):
        self._rx(self._button_msg(Buttons.UNPRESS))
        self.assertTrue(self.safety.get_controls_allowed())

  def test_cancel_button(self):
    self.safety.set_controls_allowed(1)
    self._rx(self._button_msg(Buttons.CANCEL))
    self.assertFalse(self.safety.get_controls_allowed())

  def test_brake_safety_check(self):
    for enabled in [0, 1]:
      for b in range(0, 500):
        self.safety.set_controls_allowed(enabled)
        if abs(b) > MAX_BRAKE or (not enabled and b != 0):
          self.assertFalse(self._tx(self._send_brake_msg(b)))
        else:
          self.assertTrue(self._tx(self._send_brake_msg(b)))

  def test_gas_safety_check(self):
    for enabled in [0, 1]:
      for g in range(0, 2**12 - 1):
        self.safety.set_controls_allowed(enabled)
        if abs(g) > MAX_GAS or (not enabled and g != MAX_REGEN):
          self.assertFalse(self._tx(self._send_gas_msg(g)))
        else:
          self.assertTrue(self._tx(self._send_gas_msg(g)))

  def test_tx_hook_on_pedal_pressed(self):
    for pedal in ['brake', 'gas']:
      if pedal == 'brake':
        # brake_pressed_prev and vehicle_moving
        self._rx(self._speed_msg(100))
        self._rx(self._user_brake_msg(MAX_BRAKE))
      elif pedal == 'gas':
        # gas_pressed_prev
        self._rx(self._user_gas_msg(MAX_GAS))

      self.safety.set_controls_allowed(1)
      self.assertFalse(self._tx(self._send_brake_msg(MAX_BRAKE)))
      self.assertFalse(self._tx(self._torque_cmd_msg(self.MAX_RATE_UP)))
      self.assertFalse(self._tx(self._send_gas_msg(MAX_GAS)))

      # reset status
      self.safety.set_controls_allowed(0)
      self._tx(self._send_brake_msg(0))
      self._tx(self._torque_cmd_msg(0))
      if pedal == 'brake':
        self._rx(self._speed_msg(0))
        self._rx(self._user_brake_msg(0))
      elif pedal == 'gas':
        self._rx(self._user_gas_msg(0))

  def test_tx_hook_on_pedal_pressed_on_alternative_gas_experience(self):
    for pedal in ['brake', 'gas']:
      self.safety.set_alternative_experience(ALTERNATIVE_EXPERIENCE.DISABLE_DISENGAGE_ON_GAS)
      if pedal == 'brake':
        # brake_pressed_prev and vehicle_moving
        self._rx(self._speed_msg(100))
        self._rx(self._user_brake_msg(MAX_BRAKE))
        allow_ctrl = False
      elif pedal == 'gas':
        # gas_pressed_prev
        self._rx(self._user_gas_msg(MAX_GAS))
        allow_ctrl = True

      # Test we allow lateral on gas press, but never longitudinal
      self.safety.set_controls_allowed(1)
      self.assertEqual(allow_ctrl, self._tx(self._torque_cmd_msg(self.MAX_RATE_UP)))
      self.assertFalse(self._tx(self._send_brake_msg(MAX_BRAKE)))
      self.assertFalse(self._tx(self._send_gas_msg(MAX_GAS)))

      # reset status
      if pedal == 'brake':
        self._rx(self._speed_msg(0))
        self._rx(self._user_brake_msg(0))
      elif pedal == 'gas':
        self._rx(self._user_gas_msg(0))


if __name__ == "__main__":
  unittest.main()
