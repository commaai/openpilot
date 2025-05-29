#!/usr/bin/env python3
import unittest
import numpy as np
from opendbc.car.structs import CarParams
from opendbc.safety.tests.libsafety import libsafety_py
import opendbc.safety.tests.common as common
from opendbc.safety.tests.common import CANPackerPanda
from opendbc.car.volkswagen.values import VolkswagenSafetyFlags

MAX_ACCEL = 2.0
MIN_ACCEL = -3.5

MSG_ESP_19 = 0xB2       # RX from ABS, for wheel speeds
MSG_LH_EPS_03 = 0x9F    # RX from EPS, for driver steering torque
MSG_ESP_05 = 0x106      # RX from ABS, for brake light state
MSG_TSK_06 = 0x120      # RX from ECU, for ACC status from drivetrain coordinator
MSG_MOTOR_20 = 0x121    # RX from ECU, for driver throttle input
MSG_ACC_06 = 0x122      # TX by OP, ACC control instructions to the drivetrain coordinator
MSG_HCA_01 = 0x126      # TX by OP, Heading Control Assist steering torque
MSG_GRA_ACC_01 = 0x12B  # TX by OP, ACC control buttons for cancel/resume
MSG_ACC_07 = 0x12E      # TX by OP, ACC control instructions to the drivetrain coordinator
MSG_ACC_02 = 0x30C      # TX by OP, ACC HUD data to the instrument cluster
MSG_LDW_02 = 0x397      # TX by OP, Lane line recognition and text alerts


class TestVolkswagenMqbSafetyBase(common.PandaCarSafetyTest, common.DriverTorqueSteeringSafetyTest):
  RELAY_MALFUNCTION_ADDRS = {0: (MSG_HCA_01, MSG_LDW_02), 2: (MSG_LH_EPS_03,)}

  MAX_RATE_UP = 4
  MAX_RATE_DOWN = 10
  MAX_TORQUE_LOOKUP = [0], [300]
  MAX_RT_DELTA = 75

  DRIVER_TORQUE_ALLOWANCE = 80
  DRIVER_TORQUE_FACTOR = 3

  # Wheel speeds _esp_19_msg
  def _speed_msg(self, speed):
    values = {"ESP_%s_Radgeschw_02" % s: speed for s in ["HL", "HR", "VL", "VR"]}
    return self.packer.make_can_msg_panda("ESP_19", 0, values)

  # Driver brake pressure over threshold
  def _esp_05_msg(self, brake):
    values = {"ESP_Fahrer_bremst": brake}
    return self.packer.make_can_msg_panda("ESP_05", 0, values)

  # Brake pedal switch
  def _motor_14_msg(self, brake):
    values = {"MO_Fahrer_bremst": brake}
    return self.packer.make_can_msg_panda("Motor_14", 0, values)

  def _user_brake_msg(self, brake):
    return self._motor_14_msg(brake)

  # Driver throttle input
  def _user_gas_msg(self, gas):
    values = {"MO_Fahrpedalrohwert_01": gas}
    return self.packer.make_can_msg_panda("Motor_20", 0, values)

  # ACC engagement status
  def _tsk_status_msg(self, enable, main_switch=True):
    if main_switch:
      tsk_status = 3 if enable else 2
    else:
      tsk_status = 0
    values = {"TSK_Status": tsk_status}
    return self.packer.make_can_msg_panda("TSK_06", 0, values)

  def _pcm_status_msg(self, enable):
    return self._tsk_status_msg(enable)

  # Driver steering input torque
  def _torque_driver_msg(self, torque):
    values = {"EPS_Lenkmoment": abs(torque), "EPS_VZ_Lenkmoment": torque < 0}
    return self.packer.make_can_msg_panda("LH_EPS_03", 0, values)

  # openpilot steering output torque
  def _torque_cmd_msg(self, torque, steer_req=1):
    values = {"HCA_01_LM_Offset": abs(torque), "HCA_01_LM_OffSign": torque < 0, "HCA_01_Sendestatus": steer_req}
    return self.packer.make_can_msg_panda("HCA_01", 0, values)

  # Cruise control buttons
  def _gra_acc_01_msg(self, cancel=0, resume=0, _set=0, bus=2):
    values = {"GRA_Abbrechen": cancel, "GRA_Tip_Setzen": _set, "GRA_Tip_Wiederaufnahme": resume}
    return self.packer.make_can_msg_panda("GRA_ACC_01", bus, values)

  # Acceleration request to drivetrain coordinator
  def _acc_06_msg(self, accel):
    values = {"ACC_Sollbeschleunigung_02": accel}
    return self.packer.make_can_msg_panda("ACC_06", 0, values)

  # Acceleration request to drivetrain coordinator
  def _acc_07_msg(self, accel, secondary_accel=3.02):
    values = {"ACC_Sollbeschleunigung_02": accel, "ACC_Folgebeschl": secondary_accel}
    return self.packer.make_can_msg_panda("ACC_07", 0, values)

  # Verify brake_pressed is true if either the switch or pressure threshold signals are true
  def test_redundant_brake_signals(self):
    test_combinations = [(True, True, True), (True, True, False), (True, False, True), (False, False, False)]
    for brake_pressed, motor_14_signal, esp_05_signal in test_combinations:
      self._rx(self._motor_14_msg(False))
      self._rx(self._esp_05_msg(False))
      self.assertFalse(self.safety.get_brake_pressed_prev())
      self._rx(self._motor_14_msg(motor_14_signal))
      self._rx(self._esp_05_msg(esp_05_signal))
      self.assertEqual(brake_pressed, self.safety.get_brake_pressed_prev(),
                       f"expected {brake_pressed=} with {motor_14_signal=} and {esp_05_signal=}")

  def test_torque_measurements(self):
    # TODO: make this test work with all cars
    self._rx(self._torque_driver_msg(50))
    self._rx(self._torque_driver_msg(-50))
    self._rx(self._torque_driver_msg(0))
    self._rx(self._torque_driver_msg(0))
    self._rx(self._torque_driver_msg(0))
    self._rx(self._torque_driver_msg(0))

    self.assertEqual(-50, self.safety.get_torque_driver_min())
    self.assertEqual(50, self.safety.get_torque_driver_max())

    self._rx(self._torque_driver_msg(0))
    self.assertEqual(0, self.safety.get_torque_driver_max())
    self.assertEqual(-50, self.safety.get_torque_driver_min())

    self._rx(self._torque_driver_msg(0))
    self.assertEqual(0, self.safety.get_torque_driver_max())
    self.assertEqual(0, self.safety.get_torque_driver_min())


class TestVolkswagenMqbStockSafety(TestVolkswagenMqbSafetyBase):
  TX_MSGS = [[MSG_HCA_01, 0], [MSG_LDW_02, 0], [MSG_LH_EPS_03, 2], [MSG_GRA_ACC_01, 0], [MSG_GRA_ACC_01, 2]]
  FWD_BLACKLISTED_ADDRS = {0: [MSG_LH_EPS_03], 2: [MSG_HCA_01, MSG_LDW_02]}

  def setUp(self):
    self.packer = CANPackerPanda("vw_mqb")
    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(CarParams.SafetyModel.volkswagen, 0)
    self.safety.init_tests()

  def test_spam_cancel_safety_check(self):
    self.safety.set_controls_allowed(0)
    self.assertTrue(self._tx(self._gra_acc_01_msg(cancel=1)))
    self.assertFalse(self._tx(self._gra_acc_01_msg(resume=1)))
    self.assertFalse(self._tx(self._gra_acc_01_msg(_set=1)))
    # do not block resume if we are engaged already
    self.safety.set_controls_allowed(1)
    self.assertTrue(self._tx(self._gra_acc_01_msg(resume=1)))


class TestVolkswagenMqbLongSafety(TestVolkswagenMqbSafetyBase):
  TX_MSGS = [[MSG_HCA_01, 0], [MSG_LDW_02, 0], [MSG_LH_EPS_03, 2], [MSG_ACC_02, 0], [MSG_ACC_06, 0], [MSG_ACC_07, 0]]
  FWD_BLACKLISTED_ADDRS = {0: [MSG_LH_EPS_03], 2: [MSG_HCA_01, MSG_LDW_02, MSG_ACC_02, MSG_ACC_06, MSG_ACC_07]}
  RELAY_MALFUNCTION_ADDRS = {0: (MSG_HCA_01, MSG_LDW_02, MSG_ACC_02, MSG_ACC_06, MSG_ACC_07), 2: (MSG_LH_EPS_03,)}
  INACTIVE_ACCEL = 3.01

  def setUp(self):
    self.packer = CANPackerPanda("vw_mqb")
    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(CarParams.SafetyModel.volkswagen, VolkswagenSafetyFlags.LONG_CONTROL)
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
      self._rx(self._gra_acc_01_msg(_set=(button == "set"), resume=(button == "resume"), bus=0))
      self.assertFalse(self.safety.get_controls_allowed(), f"controls allowed on {button} with main switch off")
      self._rx(self._tsk_status_msg(False, main_switch=True))
      self._rx(self._gra_acc_01_msg(_set=(button == "set"), resume=(button == "resume"), bus=0))
      self.assertFalse(self.safety.get_controls_allowed(), f"controls allowed on {button} rising edge")
      self._rx(self._gra_acc_01_msg(bus=0))
      self.assertTrue(self.safety.get_controls_allowed(), f"controls not allowed on {button} falling edge")

  def test_cancel_button(self):
    # Disable on rising edge of cancel button
    self._rx(self._tsk_status_msg(False, main_switch=True))
    self.safety.set_controls_allowed(1)
    self._rx(self._gra_acc_01_msg(cancel=True, bus=0))
    self.assertFalse(self.safety.get_controls_allowed(), "controls allowed after cancel")

  def test_main_switch(self):
    # Disable as soon as main switch turns off
    self._rx(self._tsk_status_msg(False, main_switch=True))
    self.safety.set_controls_allowed(1)
    self._rx(self._tsk_status_msg(False, main_switch=False))
    self.assertFalse(self.safety.get_controls_allowed(), "controls allowed after ACC main switch off")

  def test_accel_safety_check(self):
    for controls_allowed in [True, False]:
      # enforce we don't skip over 0 or inactive accel
      for accel in np.concatenate((np.arange(MIN_ACCEL - 2, MAX_ACCEL + 2, 0.03), [0, self.INACTIVE_ACCEL])):
        accel = round(accel, 2)  # floats might not hit exact boundary conditions without rounding
        is_inactive_accel = accel == self.INACTIVE_ACCEL
        send = (controls_allowed and MIN_ACCEL <= accel <= MAX_ACCEL) or is_inactive_accel
        self.safety.set_controls_allowed(controls_allowed)
        # primary accel request used by ECU
        self.assertEqual(send, self._tx(self._acc_06_msg(accel)), (controls_allowed, accel))
        # additional accel request used by ABS/ESP
        self.assertEqual(send, self._tx(self._acc_07_msg(accel)), (controls_allowed, accel))
        # ensure the optional secondary accel field remains inactive for now
        self.assertEqual(is_inactive_accel, self._tx(self._acc_07_msg(accel, secondary_accel=accel)), (controls_allowed, accel))


if __name__ == "__main__":
  unittest.main()
