#!/usr/bin/env python3
import unittest
import numpy as np
from panda import Panda
from panda.tests.safety import libpandasafety_py
import panda.tests.safety.common as common
from panda.tests.safety.common import CANPackerPanda, UNSAFE_MODE

MAX_RATE_UP = 7
MAX_RATE_DOWN = 17
MAX_STEER = 300
MAX_BRAKE = 350
MAX_GAS = 3072
MAX_REGEN = 1404

MAX_RT_DELTA = 128
RT_INTERVAL = 250000

DRIVER_TORQUE_ALLOWANCE = 50
DRIVER_TORQUE_FACTOR = 4

class TestGmSafety(common.PandaSafetyTest):
  TX_MSGS = [[384, 0], [1033, 0], [1034, 0], [715, 0], [880, 0],  # pt bus
             [161, 1], [774, 1], [776, 1], [784, 1],  # obs bus
             [789, 2],  # ch bus
             [0x104c006c, 3], [0x10400060]]  # gmlan
  STANDSTILL_THRESHOLD = 0
  RELAY_MALFUNCTION_ADDR = 384
  RELAY_MALFUNCTION_BUS = 0
  FWD_BLACKLISTED_ADDRS = {}
  FWD_BUS_LOOKUP = {}

  def setUp(self):
    self.packer = CANPackerPanda("gm_global_a_powertrain")
    self.packer_chassis = CANPackerPanda("gm_global_a_chassis")
    self.safety = libpandasafety_py.libpandasafety
    self.safety.set_safety_hooks(Panda.SAFETY_GM, 0)
    self.safety.init_tests_gm()

  # override these tests from PandaSafetyTest, GM uses button enable
  def test_disable_control_allowed_from_cruise(self): pass
  def test_enable_control_allowed_from_cruise(self): pass

  def _speed_msg(self, speed):
    values = {"%sWheelSpd"%s: speed for s in ["RL", "RR"]}
    return self.packer.make_can_msg_panda("EBCMWheelSpdRear", 0, values)

  def _button_msg(self, buttons):
    values = {"ACCButtons": buttons}
    return self.packer.make_can_msg_panda("ASCMSteeringButton", 0, values)

  def _brake_msg(self, brake):
    # GM safety has a brake threshold of 10
    values = {"BrakePedalPosition": 10 if brake else 0}
    return self.packer.make_can_msg_panda("EBCMBrakePedalPosition", 0, values)

  def _gas_msg(self, gas):
    values = {"AcceleratorPedal": 1 if gas else 0}
    return self.packer.make_can_msg_panda("AcceleratorPedal", 0, values)

  def _send_brake_msg(self, brake):
    values = {"FrictionBrakeCmd": -brake}
    return self.packer_chassis.make_can_msg_panda("EBCMFrictionBrakeCmd", 2, values)

  def _send_gas_msg(self, gas):
    values = {"GasRegenCmd": gas}
    return self.packer.make_can_msg_panda("ASCMGasRegenCmd", 0, values)

  def _set_prev_torque(self, t):
    self.safety.set_gm_desired_torque_last(t)
    self.safety.set_gm_rt_torque_last(t)

  def _torque_driver_msg(self, torque):
    values = {"LKADriverAppldTrq": torque}
    return self.packer.make_can_msg_panda("PSCMStatus", 0, values)

  def _torque_msg(self, torque):
    values = {"LKASteeringCmd": torque}
    return self.packer.make_can_msg_panda("ASCMLKASteeringCmd", 0, values)

  def test_resume_button(self):
    RESUME_BTN = 2
    self.safety.set_controls_allowed(0)
    self.safety.safety_rx_hook(self._button_msg(RESUME_BTN))
    self.assertTrue(self.safety.get_controls_allowed())

  def test_set_button(self):
    SET_BTN = 3
    self.safety.set_controls_allowed(0)
    self.safety.safety_rx_hook(self._button_msg(SET_BTN))
    self.assertTrue(self.safety.get_controls_allowed())

  def test_cancel_button(self):
    CANCEL_BTN = 6
    self.safety.set_controls_allowed(1)
    self.safety.safety_rx_hook(self._button_msg(CANCEL_BTN))
    self.assertFalse(self.safety.get_controls_allowed())

  def test_brake_safety_check(self):
    for enabled in [0, 1]:
      for b in range(0, 500):
        self.safety.set_controls_allowed(enabled)
        if abs(b) > MAX_BRAKE or (not enabled and b != 0):
          self.assertFalse(self.safety.safety_tx_hook(self._send_brake_msg(b)))
        else:
          self.assertTrue(self.safety.safety_tx_hook(self._send_brake_msg(b)))

  def test_gas_safety_check(self):
    for enabled in [0, 1]:
      for g in range(0, 2**12-1):
        self.safety.set_controls_allowed(enabled)
        if abs(g) > MAX_GAS or (not enabled and g != MAX_REGEN):
          self.assertFalse(self.safety.safety_tx_hook(self._send_gas_msg(g)))
        else:
          self.assertTrue(self.safety.safety_tx_hook(self._send_gas_msg(g)))

  def test_steer_safety_check(self):
    for enabled in [0, 1]:
      for t in range(-0x200, 0x200):
        self.safety.set_controls_allowed(enabled)
        self._set_prev_torque(t)
        if abs(t) > MAX_STEER or (not enabled and abs(t) > 0):
          self.assertFalse(self.safety.safety_tx_hook(self._torque_msg(t)))
        else:
          self.assertTrue(self.safety.safety_tx_hook(self._torque_msg(t)))

  def test_non_realtime_limit_up(self):
    self.safety.set_gm_torque_driver(0, 0)
    self.safety.set_controls_allowed(True)

    self._set_prev_torque(0)
    self.assertTrue(self.safety.safety_tx_hook(self._torque_msg(MAX_RATE_UP)))
    self._set_prev_torque(0)
    self.assertTrue(self.safety.safety_tx_hook(self._torque_msg(-MAX_RATE_UP)))

    self._set_prev_torque(0)
    self.assertFalse(self.safety.safety_tx_hook(self._torque_msg(MAX_RATE_UP + 1)))
    self.safety.set_controls_allowed(True)
    self._set_prev_torque(0)
    self.assertFalse(self.safety.safety_tx_hook(self._torque_msg(-MAX_RATE_UP - 1)))

  def test_non_realtime_limit_down(self):
    self.safety.set_gm_torque_driver(0, 0)
    self.safety.set_controls_allowed(True)

  def test_against_torque_driver(self):
    self.safety.set_controls_allowed(True)

    for sign in [-1, 1]:
      for t in np.arange(0, DRIVER_TORQUE_ALLOWANCE + 1, 1):
        t *= -sign
        self.safety.set_gm_torque_driver(t, t)
        self._set_prev_torque(MAX_STEER * sign)
        self.assertTrue(self.safety.safety_tx_hook(self._torque_msg(MAX_STEER * sign)))

      self.safety.set_gm_torque_driver(DRIVER_TORQUE_ALLOWANCE + 1, DRIVER_TORQUE_ALLOWANCE + 1)
      self.assertFalse(self.safety.safety_tx_hook(self._torque_msg(-MAX_STEER)))

    # spot check some individual cases
    for sign in [-1, 1]:
      driver_torque = (DRIVER_TORQUE_ALLOWANCE + 10) * sign
      torque_desired = (MAX_STEER - 10 * DRIVER_TORQUE_FACTOR) * sign
      delta = 1 * sign
      self._set_prev_torque(torque_desired)
      self.safety.set_gm_torque_driver(-driver_torque, -driver_torque)
      self.assertTrue(self.safety.safety_tx_hook(self._torque_msg(torque_desired)))
      self._set_prev_torque(torque_desired + delta)
      self.safety.set_gm_torque_driver(-driver_torque, -driver_torque)
      self.assertFalse(self.safety.safety_tx_hook(self._torque_msg(torque_desired + delta)))

      self._set_prev_torque(MAX_STEER * sign)
      self.safety.set_gm_torque_driver(-MAX_STEER * sign, -MAX_STEER * sign)
      self.assertTrue(self.safety.safety_tx_hook(self._torque_msg((MAX_STEER - MAX_RATE_DOWN) * sign)))
      self._set_prev_torque(MAX_STEER * sign)
      self.safety.set_gm_torque_driver(-MAX_STEER * sign, -MAX_STEER * sign)
      self.assertTrue(self.safety.safety_tx_hook(self._torque_msg(0)))
      self._set_prev_torque(MAX_STEER * sign)
      self.safety.set_gm_torque_driver(-MAX_STEER * sign, -MAX_STEER * sign)
      self.assertFalse(self.safety.safety_tx_hook(self._torque_msg((MAX_STEER - MAX_RATE_DOWN + 1) * sign)))


  def test_realtime_limits(self):
    self.safety.set_controls_allowed(True)

    for sign in [-1, 1]:
      self.safety.init_tests_gm()
      self._set_prev_torque(0)
      self.safety.set_gm_torque_driver(0, 0)
      for t in np.arange(0, MAX_RT_DELTA, 1):
        t *= sign
        self.assertTrue(self.safety.safety_tx_hook(self._torque_msg(t)))
      self.assertFalse(self.safety.safety_tx_hook(self._torque_msg(sign * (MAX_RT_DELTA + 1))))

      self._set_prev_torque(0)
      for t in np.arange(0, MAX_RT_DELTA, 1):
        t *= sign
        self.assertTrue(self.safety.safety_tx_hook(self._torque_msg(t)))

      # Increase timer to update rt_torque_last
      self.safety.set_timer(RT_INTERVAL + 1)
      self.assertTrue(self.safety.safety_tx_hook(self._torque_msg(sign * (MAX_RT_DELTA - 1))))
      self.assertTrue(self.safety.safety_tx_hook(self._torque_msg(sign * (MAX_RT_DELTA + 1))))


  def test_tx_hook_on_pedal_pressed(self):
    for pedal in ['brake', 'gas']:
      if pedal == 'brake':
        # brake_pressed_prev and honda_moving
        self.safety.safety_rx_hook(self._speed_msg(100))
        self.safety.safety_rx_hook(self._brake_msg(MAX_BRAKE))
      elif pedal == 'gas':
        # gas_pressed_prev
        self.safety.safety_rx_hook(self._gas_msg(MAX_GAS))

      self.safety.set_controls_allowed(1)
      self.assertFalse(self.safety.safety_tx_hook(self._send_brake_msg(MAX_BRAKE)))
      self.assertFalse(self.safety.safety_tx_hook(self._torque_msg(MAX_RATE_UP)))
      self.assertFalse(self.safety.safety_tx_hook(self._send_gas_msg(MAX_GAS)))

      # reset status
      self.safety.set_controls_allowed(0)
      self.safety.safety_tx_hook(self._send_brake_msg(0))
      self.safety.safety_tx_hook(self._torque_msg(0))
      if pedal == 'brake':
        self.safety.safety_rx_hook(self._speed_msg(0))
        self.safety.safety_rx_hook(self._brake_msg(0))
      elif pedal == 'gas':
        self.safety.safety_rx_hook(self._gas_msg(0))

  def test_tx_hook_on_pedal_pressed_on_unsafe_gas_mode(self):
    for pedal in ['brake', 'gas']:
      self.safety.set_unsafe_mode(UNSAFE_MODE.DISABLE_DISENGAGE_ON_GAS)
      if pedal == 'brake':
        # brake_pressed_prev and honda_moving
        self.safety.safety_rx_hook(self._speed_msg(100))
        self.safety.safety_rx_hook(self._brake_msg(MAX_BRAKE))
        allow_ctrl = False
      elif pedal == 'gas':
        # gas_pressed_prev
        self.safety.safety_rx_hook(self._gas_msg(MAX_GAS))
        allow_ctrl = True

      self.safety.set_controls_allowed(1)
      self.assertEqual(allow_ctrl, self.safety.safety_tx_hook(self._send_brake_msg(MAX_BRAKE)))
      self.assertEqual(allow_ctrl, self.safety.safety_tx_hook(self._torque_msg(MAX_RATE_UP)))
      self.assertEqual(allow_ctrl, self.safety.safety_tx_hook(self._send_gas_msg(MAX_GAS)))

      # reset status
      self.safety.set_controls_allowed(0)
      self.safety.set_unsafe_mode(UNSAFE_MODE.DEFAULT)
      self.safety.safety_tx_hook(self._send_brake_msg(0))
      self.safety.safety_tx_hook(self._torque_msg(0))
      if pedal == 'brake':
        self.safety.safety_rx_hook(self._speed_msg(0))
        self.safety.safety_rx_hook(self._brake_msg(0))
      elif pedal == 'gas':
        self.safety.safety_rx_hook(self._gas_msg(0))

if __name__ == "__main__":
  unittest.main()
