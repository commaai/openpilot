#!/usr/bin/env python3
import unittest
import numpy as np
from panda import Panda
import panda.tests.safety.common as common
from panda.tests.libpanda import libpanda_py
from panda.tests.safety.common import CANPackerPanda

MAX_ACCEL = 2.0
MIN_ACCEL = -3.5


class CONTROL_LEVER_STATE:
  DN_1ST = 32
  UP_1ST = 16
  DN_2ND = 8
  UP_2ND = 4
  RWD = 2
  FWD = 1
  IDLE = 0


class TestTeslaSafety(common.PandaCarSafetyTest):
  STANDSTILL_THRESHOLD = 0
  GAS_PRESSED_THRESHOLD = 3
  FWD_BUS_LOOKUP = {0: 2, 2: 0}

  def setUp(self):
    self.packer = None
    raise unittest.SkipTest

  def _speed_msg(self, speed):
    values = {"DI_vehicleSpeed": speed / 0.447}
    return self.packer.make_can_msg_panda("DI_torque2", 0, values)

  def _user_brake_msg(self, brake):
    values = {"driverBrakeStatus": 2 if brake else 1}
    return self.packer.make_can_msg_panda("BrakeMessage", 0, values)

  def _user_gas_msg(self, gas):
    values = {"DI_pedalPos": gas}
    return self.packer.make_can_msg_panda("DI_torque1", 0, values)

  def _control_lever_cmd(self, command):
    values = {"SpdCtrlLvr_Stat": command}
    return self.packer.make_can_msg_panda("STW_ACTN_RQ", 0, values)

  def _pcm_status_msg(self, enable):
    values = {"DI_cruiseState": 2 if enable else 0}
    return self.packer.make_can_msg_panda("DI_state", 0, values)

  def _long_control_msg(self, set_speed, acc_val=0, jerk_limits=(0, 0), accel_limits=(0, 0), aeb_event=0, bus=0):
    values = {
      "DAS_setSpeed": set_speed,
      "DAS_accState": acc_val,
      "DAS_aebEvent": aeb_event,
      "DAS_jerkMin": jerk_limits[0],
      "DAS_jerkMax": jerk_limits[1],
      "DAS_accelMin": accel_limits[0],
      "DAS_accelMax": accel_limits[1],
    }
    return self.packer.make_can_msg_panda("DAS_control", bus, values)


class TestTeslaSteeringSafety(TestTeslaSafety, common.AngleSteeringSafetyTest):
  TX_MSGS = [[0x488, 0], [0x45, 0], [0x45, 2]]
  RELAY_MALFUNCTION_ADDRS = {0: (0x488,)}
  FWD_BLACKLISTED_ADDRS = {2: [0x488]}

  # Angle control limits
  DEG_TO_CAN = 10

  ANGLE_RATE_BP = [0., 5., 15.]
  ANGLE_RATE_UP = [10., 1.6, .3]  # windup limit
  ANGLE_RATE_DOWN = [10., 7.0, .8]  # unwind limit

  def setUp(self):
    self.packer = CANPackerPanda("tesla_can")
    self.safety = libpanda_py.libpanda
    self.safety.set_safety_hooks(Panda.SAFETY_TESLA, 0)
    self.safety.init_tests()

  def _angle_cmd_msg(self, angle: float, enabled: bool):
    values = {"DAS_steeringAngleRequest": angle, "DAS_steeringControlType": 1 if enabled else 0}
    return self.packer.make_can_msg_panda("DAS_steeringControl", 0, values)

  def _angle_meas_msg(self, angle: float):
    values = {"EPAS_internalSAS": angle}
    return self.packer.make_can_msg_panda("EPAS_sysStatus", 0, values)

  def test_acc_buttons(self):
    """
      FWD (cancel) always allowed.
    """
    btns = [
      (CONTROL_LEVER_STATE.FWD, True),
      (CONTROL_LEVER_STATE.RWD, False),
      (CONTROL_LEVER_STATE.UP_1ST, False),
      (CONTROL_LEVER_STATE.UP_2ND, False),
      (CONTROL_LEVER_STATE.DN_1ST, False),
      (CONTROL_LEVER_STATE.DN_2ND, False),
      (CONTROL_LEVER_STATE.IDLE, False),
    ]
    for btn, should_tx in btns:
      for controls_allowed in (True, False):
        self.safety.set_controls_allowed(controls_allowed)
        tx = self._tx(self._control_lever_cmd(btn))
        self.assertEqual(tx, should_tx)


class TestTeslaRavenSteeringSafety(TestTeslaSteeringSafety):
  def setUp(self):
    self.packer = CANPackerPanda("tesla_can")
    self.safety = libpanda_py.libpanda
    self.safety.set_safety_hooks(Panda.SAFETY_TESLA, Panda.FLAG_TESLA_RAVEN)
    self.safety.init_tests()

  def _angle_meas_msg(self, angle: float):
    values = {"EPAS_internalSAS": angle}
    return self.packer.make_can_msg_panda("EPAS3P_sysStatus", 2, values)

class TestTeslaLongitudinalSafety(TestTeslaSafety):
  def setUp(self):
    raise unittest.SkipTest

  def test_no_aeb(self):
    for aeb_event in range(4):
      self.assertEqual(self._tx(self._long_control_msg(10, aeb_event=aeb_event)), aeb_event == 0)

  def test_stock_aeb_passthrough(self):
    no_aeb_msg = self._long_control_msg(10, aeb_event=0)
    no_aeb_msg_cam = self._long_control_msg(10, aeb_event=0, bus=2)
    aeb_msg_cam = self._long_control_msg(10, aeb_event=1, bus=2)

    # stock system sends no AEB -> no forwarding, and OP is allowed to TX
    self.assertEqual(1, self._rx(no_aeb_msg_cam))
    self.assertEqual(-1, self.safety.safety_fwd_hook(2, no_aeb_msg_cam.addr))
    self.assertEqual(True, self._tx(no_aeb_msg))

    # stock system sends AEB -> forwarding, and OP is not allowed to TX
    self.assertEqual(1, self._rx(aeb_msg_cam))
    self.assertEqual(0, self.safety.safety_fwd_hook(2, aeb_msg_cam.addr))
    self.assertEqual(False, self._tx(no_aeb_msg))

  def test_acc_accel_limits(self):
    for controls_allowed in [True, False]:
      self.safety.set_controls_allowed(controls_allowed)
      for min_accel in np.arange(MIN_ACCEL - 1, MAX_ACCEL + 1, 0.1):
        for max_accel in np.arange(MIN_ACCEL - 1, MAX_ACCEL + 1, 0.1):
          # floats might not hit exact boundary conditions without rounding
          min_accel = round(min_accel, 2)
          max_accel = round(max_accel, 2)
          if controls_allowed:
            send = (MIN_ACCEL <= min_accel <= MAX_ACCEL) and (MIN_ACCEL <= max_accel <= MAX_ACCEL)
          else:
            send = np.all(np.isclose([min_accel, max_accel], 0, atol=0.0001))
          self.assertEqual(send, self._tx(self._long_control_msg(10, acc_val=4, accel_limits=[min_accel, max_accel])))


class TestTeslaChassisLongitudinalSafety(TestTeslaLongitudinalSafety):
  TX_MSGS = [[0x488, 0], [0x45, 0], [0x45, 2], [0x2B9, 0]]
  RELAY_MALFUNCTION_ADDRS = {0: (0x488,)}
  FWD_BLACKLISTED_ADDRS = {2: [0x2B9, 0x488]}

  def setUp(self):
    self.packer = CANPackerPanda("tesla_can")
    self.safety = libpanda_py.libpanda
    self.safety.set_safety_hooks(Panda.SAFETY_TESLA, Panda.FLAG_TESLA_LONG_CONTROL)
    self.safety.init_tests()


class TestTeslaPTLongitudinalSafety(TestTeslaLongitudinalSafety):
  TX_MSGS = [[0x2BF, 0]]
  RELAY_MALFUNCTION_ADDRS = {0: (0x2BF,)}
  FWD_BLACKLISTED_ADDRS = {2: [0x2BF]}

  def setUp(self):
    self.packer = CANPackerPanda("tesla_powertrain")
    self.safety = libpanda_py.libpanda
    self.safety.set_safety_hooks(Panda.SAFETY_TESLA, Panda.FLAG_TESLA_LONG_CONTROL | Panda.FLAG_TESLA_POWERTRAIN)
    self.safety.init_tests()


if __name__ == "__main__":
  unittest.main()
