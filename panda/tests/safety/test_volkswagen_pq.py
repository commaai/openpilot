#!/usr/bin/env python3
import unittest
from panda import Panda
from panda.tests.libpanda import libpanda_py
import panda.tests.safety.common as common
from panda.tests.safety.common import CANPackerPanda

MSG_LENKHILFE_3 = 0x0D0       # RX from EPS, for steering angle and driver steering torque
MSG_HCA_1 = 0x0D2             # TX by OP, Heading Control Assist steering torque
MSG_BREMSE_1 = 0x1A0          # RX from ABS, for ego speed
MSG_MOTOR_2 = 0x288           # RX from ECU, for CC state and brake switch state
MSG_ACC_SYSTEM = 0x368        # TX by OP, longitudinal acceleration controls
MSG_MOTOR_3 = 0x380           # RX from ECU, for driver throttle input
MSG_GRA_NEU = 0x38A           # TX by OP, ACC control buttons for cancel/resume
MSG_MOTOR_5 = 0x480           # RX from ECU, for ACC main switch state
MSG_ACC_GRA_ANZEIGE = 0x56A   # TX by OP, ACC HUD
MSG_LDW_1 = 0x5BE             # TX by OP, Lane line recognition and text alerts


class TestVolkswagenPqSafety(common.PandaCarSafetyTest, common.DriverTorqueSteeringSafetyTest):
  cruise_engaged = False

  STANDSTILL_THRESHOLD = 0
  RELAY_MALFUNCTION_ADDRS = {0: (MSG_HCA_1,)}

  MAX_RATE_UP = 6
  MAX_RATE_DOWN = 10
  MAX_TORQUE = 300
  MAX_RT_DELTA = 113
  RT_INTERVAL = 250000

  DRIVER_TORQUE_ALLOWANCE = 80
  DRIVER_TORQUE_FACTOR = 3

  @classmethod
  def setUpClass(cls):
    if cls.__name__ == "TestVolkswagenPqSafety":
      cls.packer = None
      cls.safety = None
      raise unittest.SkipTest

  def _set_prev_torque(self, t):
    self.safety.set_desired_torque_last(t)
    self.safety.set_rt_torque_last(t)

  # Ego speed (Bremse_1)
  def _speed_msg(self, speed):
    values = {"Geschwindigkeit_neu__Bremse_1_": speed}
    return self.packer.make_can_msg_panda("Bremse_1", 0, values)

  # Brake light switch (shared message Motor_2)
  def _user_brake_msg(self, brake):
    # since this signal is used for engagement status, preserve current state
    return self._motor_2_msg(brake_pressed=brake, cruise_engaged=self.safety.get_controls_allowed())

  # ACC engaged status (shared message Motor_2)
  def _pcm_status_msg(self, enable):
    self.__class__.cruise_engaged = enable
    return self._motor_2_msg(cruise_engaged=enable)

  # Acceleration request to drivetrain coordinator
  def _accel_msg(self, accel):
    values = {"ACS_Sollbeschl": accel}
    return self.packer.make_can_msg_panda("ACC_System", 0, values)

  # Driver steering input torque
  def _torque_driver_msg(self, torque):
    values = {"LH3_LM": abs(torque), "LH3_LMSign": torque < 0}
    return self.packer.make_can_msg_panda("Lenkhilfe_3", 0, values)

  # openpilot steering output torque
  def _torque_cmd_msg(self, torque, steer_req=1, hca_status=5):
    values = {"LM_Offset": abs(torque), "LM_OffSign": torque < 0, "HCA_Status": hca_status if steer_req else 3}
    return self.packer.make_can_msg_panda("HCA_1", 0, values)

  # ACC engagement and brake light switch status
  # Called indirectly for compatibility with common.py tests
  def _motor_2_msg(self, brake_pressed=False, cruise_engaged=False):
    values = {"Bremslichtschalter": brake_pressed,
              "GRA_Status": cruise_engaged}
    return self.packer.make_can_msg_panda("Motor_2", 0, values)

  # ACC main switch status
  def _motor_5_msg(self, main_switch=False):
    values = {"GRA_Hauptschalter": main_switch}
    return self.packer.make_can_msg_panda("Motor_5", 0, values)

  # Driver throttle input (Motor_3)
  def _user_gas_msg(self, gas):
    values = {"Fahrpedal_Rohsignal": gas}
    return self.packer.make_can_msg_panda("Motor_3", 0, values)

  # Cruise control buttons (GRA_Neu)
  def _button_msg(self, _set=False, resume=False, cancel=False, bus=2):
    values = {"GRA_Neu_Setzen": _set, "GRA_Recall": resume, "GRA_Abbrechen": cancel}
    return self.packer.make_can_msg_panda("GRA_Neu", bus, values)

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


class TestVolkswagenPqStockSafety(TestVolkswagenPqSafety):
  # Transmit of GRA_Neu is allowed on bus 0 and 2 to keep compatibility with gateway and camera integration
  TX_MSGS = [[MSG_HCA_1, 0], [MSG_GRA_NEU, 0], [MSG_GRA_NEU, 2], [MSG_LDW_1, 0]]
  FWD_BLACKLISTED_ADDRS = {2: [MSG_HCA_1, MSG_LDW_1]}
  FWD_BUS_LOOKUP = {0: 2, 2: 0}

  def setUp(self):
    self.packer = CANPackerPanda("vw_golf_mk4")
    self.safety = libpanda_py.libpanda
    self.safety.set_safety_hooks(Panda.SAFETY_VOLKSWAGEN_PQ, 0)
    self.safety.init_tests()

  def test_spam_cancel_safety_check(self):
    self.safety.set_controls_allowed(0)
    self.assertTrue(self._tx(self._button_msg(cancel=True)))
    self.assertFalse(self._tx(self._button_msg(resume=True)))
    self.assertFalse(self._tx(self._button_msg(_set=True)))
    # do not block resume if we are engaged already
    self.safety.set_controls_allowed(1)
    self.assertTrue(self._tx(self._button_msg(resume=True)))


class TestVolkswagenPqLongSafety(TestVolkswagenPqSafety, common.LongitudinalAccelSafetyTest):
  TX_MSGS = [[MSG_HCA_1, 0], [MSG_LDW_1, 0], [MSG_ACC_SYSTEM, 0], [MSG_ACC_GRA_ANZEIGE, 0]]
  FWD_BLACKLISTED_ADDRS = {2: [MSG_HCA_1, MSG_LDW_1, MSG_ACC_SYSTEM, MSG_ACC_GRA_ANZEIGE]}
  FWD_BUS_LOOKUP = {0: 2, 2: 0}
  INACTIVE_ACCEL = 3.01

  def setUp(self):
    self.packer = CANPackerPanda("vw_golf_mk4")
    self.safety = libpanda_py.libpanda
    self.safety.set_safety_hooks(Panda.SAFETY_VOLKSWAGEN_PQ, Panda.FLAG_VOLKSWAGEN_LONG_CONTROL)
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
      self._rx(self._motor_5_msg(main_switch=False))
      self._rx(self._button_msg(_set=(button == "set"), resume=(button == "resume"), bus=0))
      self._rx(self._button_msg(bus=0))
      self.assertFalse(self.safety.get_controls_allowed(), f"controls allowed on {button} with main switch off")
      self._rx(self._motor_5_msg(main_switch=True))
      self._rx(self._button_msg(_set=(button == "set"), resume=(button == "resume"), bus=0))
      self.assertFalse(self.safety.get_controls_allowed(), f"controls allowed on {button} rising edge")
      self._rx(self._button_msg(bus=0))
      self.assertTrue(self.safety.get_controls_allowed(), f"controls not allowed on {button} falling edge")

  def test_cancel_button(self):
    # Disable on rising edge of cancel button
    self._rx(self._motor_5_msg(main_switch=True))
    self.safety.set_controls_allowed(1)
    self._rx(self._button_msg(cancel=True, bus=0))
    self.assertFalse(self.safety.get_controls_allowed(), "controls allowed after cancel")

  def test_main_switch(self):
    # Disable as soon as main switch turns off
    self._rx(self._motor_5_msg(main_switch=True))
    self.safety.set_controls_allowed(1)
    self._rx(self._motor_5_msg(main_switch=False))
    self.assertFalse(self.safety.get_controls_allowed(), "controls allowed after ACC main switch off")

  def test_torque_cmd_enable_variants(self):
    # The EPS rack accepts either 5 or 7 for an enabled status, with different low speed tuning behavior
    self.safety.set_controls_allowed(1)
    for enabled_status in (5, 7):
      self.assertTrue(self._tx(self._torque_cmd_msg(self.MAX_RATE_UP, steer_req=1, hca_status=enabled_status)),
                      f"torque cmd rejected with {enabled_status=}")

if __name__ == "__main__":
  unittest.main()
