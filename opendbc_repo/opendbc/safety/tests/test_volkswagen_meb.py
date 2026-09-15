#!/usr/bin/env python3
import numpy as np
import unittest

from opendbc.car.structs import CarParams
from opendbc.car.volkswagen.values import VolkswagenSafetyFlags
from opendbc.safety.tests.libsafety import libsafety_py
import opendbc.safety.tests.common as common
from opendbc.safety.tests.common import CANPackerSafety

MAX_ACCEL = 2.0
MIN_ACCEL = -3.5

# ACC_18.ACC_Anforderung_HMS
HMS_KEINE_ANFORDERUNG = 0
HMS_HALTEN = 1
HMS_ANFAHREN = 4
HMS_LOESEN_UEBER_RAMPE = 5

# ACC_18.ACC_Status_ACC
ACC_AKTIV_REGELT = 3
ACC_OVERRIDE = 4

# MEB message IDs
MSG_ACC_18     = 0x14D
MSG_KLR_01     = 0x25D
MSG_TA_01      = 0x26B
MSG_ACC_19     = 0x300
MSG_HCA_03     = 0x303
MSG_LDW_02     = 0x397


class TestVolkswagenMebSafetyBase(common.CarSafetyTest, common.CurvatureSteeringSafetyTest):
  STANDSTILL_THRESHOLD = 0

  MAX_CURVATURE = 29105
  MAX_CURVATURE_TEST = 0.195
  CURVATURE_TO_CAN = 149253.7313
  MAX_POWER = 125
  MAX_POWER_TEST = 50
  SEND_RATE = 0.02
  LATERAL_FREQUENCY = 50  # Hz

  cnt_curvature_cmd = 0

  def _set_prev_desired_power(self, power: int):
    # init with local tx sequence
    prev_allowed = self.safety.get_controls_allowed()
    self.safety.set_controls_allowed(True)
    self._tx(self._curvature_cmd_msg(0, steer_req=True, power=power))
    self.safety.set_controls_allowed(prev_allowed)

  def test_power_limit(self):
    max_power_can = self.MAX_POWER
    max_power = self.MAX_POWER_TEST
    self._set_prev_desired_power(max_power_can)
    self.safety.set_controls_allowed(True)

    self.assertTrue(self._tx(self._curvature_cmd_msg(0, steer_req=True, power=max_power)))
    self.assertTrue(self.safety.get_controls_allowed())

    self.assertFalse(self._tx(self._curvature_cmd_msg(0, steer_req=True, power=max_power + 1)))

    self.safety.set_controls_allowed(True)

    self.assertTrue(self._tx(self._curvature_cmd_msg(0, steer_req=True, power=max_power - 1)))
    self.assertTrue(self.safety.get_controls_allowed())

    self.assertFalse(self._tx(self._curvature_cmd_msg(0, steer_req=False, power=max_power)))

    self.safety.set_controls_allowed(True)

    self.assertTrue(self._tx(self._curvature_cmd_msg(0, steer_req=True, power=max_power)))
    self.assertTrue(self.safety.get_controls_allowed())

    self.assertFalse(self._tx(self._curvature_cmd_msg(0, steer_req=False, power=-max_power)))

  def test_power_without_control(self):
    max_power_can = self.MAX_POWER
    max_power = self.MAX_POWER_TEST
    self._set_prev_desired_power(max_power_can)
    self.safety.set_controls_allowed(False)

    self.assertFalse(self._tx(self._curvature_cmd_msg(0, steer_req=False, power=max_power)))
    self.assertTrue(self._tx(self._curvature_cmd_msg(0, steer_req=False, power=0)))

    self._set_prev_desired_power(max_power - 1)
    self.assertFalse(self._tx(self._curvature_cmd_msg(0, steer_req=False, power=max_power)))

    # When controls are not allowed, steer_req=True is gated only by the power check:
    # steady or rising power is disallowed, but a strictly decreasing power (steering
    # wind-down at disengagement) is permitted so the EPS torque authority ramps to zero
    self._set_prev_desired_power(max_power - 1)
    self.assertFalse(self._tx(self._curvature_cmd_msg(0, steer_req=True, power=max_power)))      # rising power
    self.assertFalse(self._tx(self._curvature_cmd_msg(0, steer_req=True, power=max_power)))      # steady power
    self.assertTrue(self._tx(self._curvature_cmd_msg(0, steer_req=True, power=max_power - 1)))   # decreasing power

  def _speed_msg(self, speed_mps: float):
    spd_kph = speed_mps * 3.6
    values = {f"{s}_Radgeschw": spd_kph for s in ("VL", "VR", "HL", "HR")}
    return self.packer.make_can_msg_safety("ESC_51", 0, values)

  def _speed_msg_2(self, speed_mps: float):
    values = {"ESP_v_Signal": speed_mps * 3.6}
    return self.packer.make_can_msg_safety("ESP_21", 0, values)

  def _motor_14_msg(self, brake):
    values = {"MO_Fahrer_bremst": brake}
    return self.packer.make_can_msg_safety("Motor_14", 0, values)

  def _user_brake_msg(self, brake):
    return self._motor_14_msg(brake)

  def _user_gas_msg(self, gas):
    values = {"Accel_Pedal_Pressure": 1 if gas else 0, "TSK_Status": 3}
    return self.packer.make_can_msg_safety("Motor_51", 0, values)

  def _vehicle_moving_msg(self, speed_mps: float):
    return self._speed_msg(speed_mps)

  def _curvature_meas_msg(self, curvature):
    values = {"Curvature": abs(curvature), "Curvature_VZ": curvature > 0}
    return self.packer.make_can_msg_safety("QFK_01", 0, values)

  def _curvature_cmd_msg(self, curvature, steer_req=1, power=50, increment_timer=True):
    if increment_timer:
      self.safety.set_timer(self.cnt_curvature_cmd * int(1e6 / self.LATERAL_FREQUENCY))
      self.__class__.cnt_curvature_cmd += 1
    values = {
      "Curvature": abs(curvature),
      "Curvature_VZ": curvature > 0,
      "RequestStatus": 4 if steer_req else 0,
      "Power": power,
    }
    return self.packer.make_can_msg_safety("HCA_03", 0, values)

  def _accel_msg(self, accel, hold_type=0, acc_status=0, acc_anfahren=0, acc_anhalten=0):
    values = {"ACC_Sollbeschleunigung_02": accel, "ACC_Anforderung_HMS": hold_type,
              "ACC_Status_ACC": acc_status, "ACC_Anfahren": acc_anfahren,
              "ACC_Anhalten": acc_anhalten}
    return self.packer.make_can_msg_safety("ACC_18", 0, values)

  def _tsk_status_msg(self, enable, main_switch=True):
    if main_switch:
      tsk_status = 3 if enable else 2
    else:
      tsk_status = 0
    values = {"TSK_Status": tsk_status}
    return self.packer.make_can_msg_safety("Motor_51", 0, values)

  def _pcm_status_msg(self, enable):
    return self._tsk_status_msg(enable)

  def _torque_driver_msg(self, torque):
    values = {"EPS_Lenkmoment": abs(torque), "EPS_VZ_Lenkmoment": torque < 0}
    return self.packer.make_can_msg_safety("LH_EPS_03", 0, values)

  def _button_msg(self, cancel=0, resume=0, _set=0, bus=2):
    values = {"GRA_Abbrechen": cancel, "GRA_Tip_Setzen": _set, "GRA_Tip_Wiederaufnahme": resume}
    return self.packer.make_can_msg_safety("GRA_ACC_01", bus, values)

  def test_curvature_measurements(self):
    self._rx(self._curvature_meas_msg(0.15))
    self._rx(self._curvature_meas_msg(-0.1))
    self._rx(self._curvature_meas_msg(0))
    self._rx(self._curvature_meas_msg(0))
    self._rx(self._curvature_meas_msg(0))
    self._rx(self._curvature_meas_msg(0))

    self.assertEqual(int(-0.1 * self.CURVATURE_TO_CAN), self.safety.get_curvature_meas_min())
    self.assertEqual(int(0.15 * self.CURVATURE_TO_CAN), self.safety.get_curvature_meas_max())

    self._rx(self._curvature_meas_msg(0))
    self.assertEqual(0, self.safety.get_curvature_meas_max())
    self.assertEqual(int(-0.1 * self.CURVATURE_TO_CAN), self.safety.get_curvature_meas_min())

    self._rx(self._curvature_meas_msg(0))
    self.assertEqual(0, self.safety.get_curvature_meas_max())
    self.assertEqual(0, self.safety.get_curvature_meas_min())

  def test_brake_signal(self):
    self._rx(self._user_brake_msg(False))
    self.assertFalse(self.safety.get_brake_pressed_prev())
    self._rx(self._user_brake_msg(True))
    self.assertTrue(self.safety.get_brake_pressed_prev())

  def test_torque_driver_measurements(self):
    for t in (0, 100, -100, 250, -250):
      self._rx(self._torque_driver_msg(t))

  def test_rx_hook(self):
    # checksum and counter checks
    for name in ("LH_EPS_03", "Motor_14", "GRA_ACC_01", "QFK_01", "ESP_21", "Motor_51", "ESC_51"):
      with self.subTest(msg=name):
        # an expected counter sequence is always accepted, and clears the wrong counter count
        next_counter = common.MAX_WRONG_COUNTERS + 1
        for counter in range(next_counter):
          self.assertTrue(self._rx(self.packer.make_can_msg_safety(name, 0, {"COUNTER": counter})))

        # mess with the checksum to make it fail, it's the first byte of every MEB message
        msg = self.packer.make_can_msg_safety(name, 0, {"COUNTER": next_counter})
        msg[0].data[0] ^= 0xFF
        self.assertFalse(self._rx(msg))

        # a stuck counter fails as well, its checksum is still correct
        for i in range(common.MAX_WRONG_COUNTERS):
          should_rx = i < common.MAX_WRONG_COUNTERS - 1
          self.assertEqual(should_rx, self._rx(self.packer.make_can_msg_safety(name, 0, {"COUNTER": next_counter})))

  def test_main_switch_off_disables_controls(self):
    self.safety.set_controls_allowed(True)
    self._rx(self._tsk_status_msg(False, main_switch=False))
    self.assertFalse(self.safety.get_controls_allowed())

  def test_cancel_button_rising_edge(self):
    self.safety.set_controls_allowed(True)
    self._rx(self._button_msg(cancel=1, bus=0))
    self.assertFalse(self.safety.get_controls_allowed())

  def test_rx_hook_speed_mismatch(self):
    for speed in np.arange(0, 40, 0.5):
      for speed_delta in np.arange(-5, 5, 0.1):
        speed_2 = round(max(speed + speed_delta, 0), 1)
        self._rx(self._speed_msg(speed))
        self._rx(self._speed_msg_2(speed_2))
        self.safety.set_controls_allowed(True)
        self._tx(self._curvature_cmd_msg(0, steer_req=True))

        within_delta = abs(speed - speed_2) <= common.MAX_SPEED_DELTA
        self.assertEqual(self.safety.get_controls_allowed(), within_delta)

  def test_curvature_violation(self):
    # if violation occurs, desired_curvature_last is reset to 0
    meas = self.MAX_CURVATURE_TEST / 4
    self.safety.set_controls_allowed(True)
    self._reset_curvature_measurement(meas)
    self._set_prev_desired_curvature(0)

    # cause a violation by sending a command far from prev=0
    self.assertFalse(self._tx(self._curvature_cmd_msg(self.MAX_CURVATURE_TEST, steer_req=True, power=50)))

    # prev should be reset to 0
    self.assertEqual(0, self.safety.get_desired_curvature_last())

  def test_curvature_cmd_when_not_steering(self):
    # Tests that only a zero curvature is allowed while the steer
    # actuation bit is 0, regardless of controls allowed or meas
    for controls_allowed in (True, False):
      self.safety.set_controls_allowed(controls_allowed)

      for steer_req in (True, False):
        for curvature_meas in np.arange(-self.MAX_CURVATURE_TEST, self.MAX_CURVATURE_TEST, self.MAX_CURVATURE_TEST / 5):
          self._reset_curvature_measurement(curvature_meas)

          for curvature_cmd in np.arange(-self.MAX_CURVATURE_TEST, self.MAX_CURVATURE_TEST, self.MAX_CURVATURE_TEST / 5):
            self._set_prev_desired_curvature(curvature_cmd)

            # controls_allowed is checked if actuation bit is 1, else the curvature must be zero (inactive)
            should_tx = controls_allowed if steer_req else round(curvature_cmd * self.CURVATURE_TO_CAN) == 0
            self.assertEqual(should_tx, self._tx(self._curvature_cmd_msg(curvature_cmd, steer_req=steer_req, power=50 if steer_req else 0)))


class TestVolkswagenMebSafety(TestVolkswagenMebSafetyBase):
  TX_MSGS = [[MSG_HCA_03, 0], [MSG_LDW_02, 0], [MSG_ACC_19, 0], [MSG_ACC_18, 0],
             [MSG_TA_01, 0], [MSG_KLR_01, 0], [MSG_KLR_01, 2]]
  FWD_BLACKLISTED_ADDRS = {0: [MSG_KLR_01],
                           2: [MSG_HCA_03, MSG_LDW_02, MSG_ACC_19, MSG_ACC_18, MSG_TA_01]}
  RELAY_MALFUNCTION_ADDRS = {0: (MSG_HCA_03, MSG_LDW_02, MSG_ACC_19, MSG_ACC_18, MSG_TA_01),
                             2: (MSG_KLR_01,)}

  ACCEL_OVERRIDE = 0
  INACTIVE_ACCEL = 3.01

  def setUp(self):
    self.packer = CANPackerSafety("vw_meb_generated")
    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(CarParams.SafetyModel.volkswagenMeb, 0)
    self.safety.init_tests()

  # stock cruise controls are entirely bypassed under openpilot longitudinal control
  def test_disable_control_allowed_from_cruise(self):
    pass

  def test_enable_control_allowed_from_cruise(self):
    pass

  def test_cruise_engaged_prev(self):
    pass

  def test_set_and_resume_buttons(self):
    for button in ["set", "resume"]:
      # ACC main switch must be on, engage on falling edge
      self.safety.set_controls_allowed(0)
      self._rx(self._tsk_status_msg(False, main_switch=False))
      self._rx(self._button_msg(_set=(button == "set"), resume=(button == "resume"), bus=0))
      self.assertFalse(self.safety.get_controls_allowed(), f"controls allowed on {button} with main switch off")
      self._rx(self._tsk_status_msg(False, main_switch=True))
      self._rx(self._button_msg(_set=(button == "set"), resume=(button == "resume"), bus=0))
      self.assertFalse(self.safety.get_controls_allowed(), f"controls allowed on {button} rising edge")
      self._rx(self._button_msg(bus=0))
      self.assertTrue(self.safety.get_controls_allowed(), f"controls not allowed on {button} falling edge")

  def test_cancel_button(self):
    # Disable on rising edge of cancel button
    self._rx(self._tsk_status_msg(False, main_switch=True))
    self.safety.set_controls_allowed(1)
    self._rx(self._button_msg(cancel=True, bus=0))
    self.assertFalse(self.safety.get_controls_allowed(), "controls allowed after cancel")

  def test_main_switch(self):
    # Disable as soon as main switch turns off
    self._rx(self._tsk_status_msg(False, main_switch=True))
    self.safety.set_controls_allowed(1)
    self._rx(self._tsk_status_msg(False, main_switch=False))
    self.assertFalse(self.safety.get_controls_allowed(), "controls allowed after ACC main switch off")

  def test_accel_safety_check(self):
    for controls_allowed in [True, False]:
      for accel in np.concatenate((np.arange(MIN_ACCEL - 2, MAX_ACCEL + 2, 0.03), [0, self.INACTIVE_ACCEL])):
        accel = round(accel, 2)
        is_inactive_accel = accel == self.INACTIVE_ACCEL
        send = (controls_allowed and MIN_ACCEL <= accel <= MAX_ACCEL) or is_inactive_accel
        self.safety.set_controls_allowed(controls_allowed)
        self.assertEqual(send, self._tx(self._accel_msg(accel)), (controls_allowed, accel))

  def test_accel_override_with_gas(self):
    self.safety.set_controls_allowed(True)
    self.safety.set_gas_pressed_prev(True)
    self.assertTrue(self._tx(self._accel_msg(self.ACCEL_OVERRIDE)))
    self.assertFalse(self._tx(self._accel_msg(MAX_ACCEL)))

  def test_hold_type_safety_check(self):
    # PARKEN engages the EPB and HALTEN holds the car, both with ACC disengaged. KEINE_ANFORDERUNG and
    # LOESEN_UEBER_RAMPE stay unconditional, the disengage ramp sends the latter once disallowed
    for controls_allowed in (True, False):
      for hold_type in range(8):
        self.safety.set_controls_allowed(controls_allowed)
        send = (hold_type in (HMS_KEINE_ANFORDERUNG, HMS_LOESEN_UEBER_RAMPE) or
                (controls_allowed and hold_type in (HMS_HALTEN, HMS_ANFAHREN)))
        self.assertEqual(send, self._tx(self._accel_msg(self.INACTIVE_ACCEL, hold_type=hold_type)))

  def test_drive_off_and_hold_request_safety_check(self):
    # ACC_Anfahren and ACC_Anhalten carry the same drive off and hold requests as the hold type
    for controls_allowed in (True, False):
      for acc_anfahren in (0, 1):
        for acc_anhalten in (0, 1):
          self.safety.set_controls_allowed(controls_allowed)
          send = controls_allowed or not (acc_anfahren or acc_anhalten)
          self.assertEqual(send, self._tx(self._accel_msg(self.INACTIVE_ACCEL, acc_anfahren=acc_anfahren,
                                                          acc_anhalten=acc_anhalten)))

  def test_acc_status_safety_check(self):
    # claiming ACC_AKTIV_REGELT or ACC_OVERRIDE is what makes the drivetrain act on our requests.
    # standby, off and the fault state stay available for the HUD with controls disallowed
    for controls_allowed in (True, False):
      for acc_status in range(8):
        self.safety.set_controls_allowed(controls_allowed)
        send = controls_allowed or acc_status not in (ACC_AKTIV_REGELT, ACC_OVERRIDE)
        self.assertEqual(send, self._tx(self._accel_msg(self.INACTIVE_ACCEL, acc_status=acc_status)))


class TestVolkswagenMebGen2Safety(TestVolkswagenMebSafety):
  def setUp(self):
    self.packer = CANPackerSafety("vw_meb_2024_generated")
    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(CarParams.SafetyModel.volkswagenMeb, VolkswagenSafetyFlags.MEB_ALT_CRC)
    self.safety.init_tests()


# ZAS_Kl_15=1
class TestVolkswagenMebIgnition(unittest.TestCase):
  TX_MSGS: list = []

  def setUp(self):
    self.safety = libsafety_py.libsafety
    self.safety.init_tests()
    self.packer = CANPackerSafety("vw_meb_generated")

  def _msg(self, counter, ign):
    return self.packer.make_can_msg_safety("Klemmen_Status_01", 0,
                                           {"Klemmen_Status_01_BZ": counter,
                                            "ZAS_Kl_15": ign})

  def test_ignition_on(self):
    for i in range(16):
      self.safety.init_tests()
      self.safety.ignition_can_hook(self._msg(i, 1))
      self.assertFalse(self.safety.get_ignition_can())
      self.safety.ignition_can_hook(self._msg((i + 1) % 16, 1))
      self.assertTrue(self.safety.get_ignition_can())

  def test_ignition_off(self):
    self.safety.ignition_can_hook(self._msg(0, 1))
    self.safety.ignition_can_hook(self._msg(1, 1))
    self.assertTrue(self.safety.get_ignition_can())
    self.safety.ignition_can_hook(self._msg(2, 0))
    self.safety.ignition_can_hook(self._msg(3, 0))
    self.assertFalse(self.safety.get_ignition_can())


if __name__ == "__main__":
  unittest.main()
