#!/usr/bin/env python3
import unittest
from panda import Panda
from panda.tests.safety import libpandasafety_py
import panda.tests.safety.common as common
from panda.tests.safety.common import CANPackerPanda, MAX_WRONG_COUNTERS

MSG_ESP_19 = 0xB2       # RX from ABS, for wheel speeds
MSG_LH_EPS_03 = 0x9F    # RX from EPS, for driver steering torque
MSG_ESP_05 = 0x106      # RX from ABS, for brake light state
MSG_TSK_06 = 0x120      # RX from ECU, for ACC status from drivetrain coordinator
MSG_MOTOR_20 = 0x121    # RX from ECU, for driver throttle input
MSG_HCA_01 = 0x126      # TX by OP, Heading Control Assist steering torque
MSG_GRA_ACC_01 = 0x12B  # TX by OP, ACC control buttons for cancel/resume
MSG_LDW_02 = 0x397      # TX by OP, Lane line recognition and text alerts


class TestVolkswagenMqbSafety(common.PandaSafetyTest, common.DriverTorqueSteeringSafetyTest):
  cnt_lh_eps_03 = 0
  cnt_esp_05 = 0
  cnt_tsk_06 = 0
  cnt_motor_20 = 0
  cnt_hca_01 = 0
  cnt_gra_acc_01 = 0

  STANDSTILL_THRESHOLD = 1
  RELAY_MALFUNCTION_ADDR = MSG_HCA_01
  RELAY_MALFUNCTION_BUS = 0

  MAX_RATE_UP = 4
  MAX_RATE_DOWN = 10
  MAX_TORQUE = 300
  MAX_RT_DELTA = 75
  RT_INTERVAL = 250000

  DRIVER_TORQUE_ALLOWANCE = 80
  DRIVER_TORQUE_FACTOR = 3

  @classmethod
  def setUpClass(cls):
    if cls.__name__ == "TestVolkswagenMqbSafety":
      cls.packer = None
      cls.safety = None
      raise unittest.SkipTest

  # Wheel speeds _esp_19_msg
  def _speed_msg(self, speed):
    values = {"ESP_%s_Radgeschw_02" % s: speed for s in ["HL", "HR", "VL", "VR"]}
    return self.packer.make_can_msg_panda("ESP_19", 0, values)

  # Brake light switch _esp_05_msg
  def _user_brake_msg(self, brake):
    values = {"ESP_Fahrer_bremst": brake, "COUNTER": self.cnt_esp_05 % 16}
    self.__class__.cnt_esp_05 += 1
    return self.packer.make_can_msg_panda("ESP_05", 0, values)

  # Driver throttle input
  def _user_gas_msg(self, gas):
    values = {"MO_Fahrpedalrohwert_01": gas, "COUNTER": self.cnt_motor_20 % 16}
    self.__class__.cnt_motor_20 += 1
    return self.packer.make_can_msg_panda("Motor_20", 0, values)

  # ACC engagement status
  def _pcm_status_msg(self, enable):
    values = {"TSK_Status": 3 if enable else 1, "COUNTER": self.cnt_tsk_06 % 16}
    self.__class__.cnt_tsk_06 += 1
    return self.packer.make_can_msg_panda("TSK_06", 0, values)

  # Driver steering input torque
  def _torque_driver_msg(self, torque):
    values = {"EPS_Lenkmoment": abs(torque), "EPS_VZ_Lenkmoment": torque < 0,
              "COUNTER": self.cnt_lh_eps_03 % 16}
    self.__class__.cnt_lh_eps_03 += 1
    return self.packer.make_can_msg_panda("LH_EPS_03", 0, values)

  # openpilot steering output torque
  def _torque_cmd_msg(self, torque, steer_req=1):
    values = {"Assist_Torque": abs(torque), "Assist_VZ": torque < 0,
              "COUNTER": self.cnt_hca_01 % 16}
    self.__class__.cnt_hca_01 += 1
    return self.packer.make_can_msg_panda("HCA_01", 0, values)

  # Cruise control buttons
  def _gra_acc_01_msg(self, cancel=0, resume=0, _set=0):
    values = {"GRA_Abbrechen": cancel, "GRA_Tip_Setzen": _set,
              "GRA_Tip_Wiederaufnahme": resume, "COUNTER": self.cnt_gra_acc_01 % 16}
    self.__class__.cnt_gra_acc_01 += 1
    return self.packer.make_can_msg_panda("GRA_ACC_01", 0, values)

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

  def test_rx_hook(self):
    # checksum checks
    # TODO: Would be ideal to check ESP_19 as well, but it has no checksum
    # or counter, and I'm not sure if we can easily validate Panda's simple
    # temporal reception-rate check here.
    for msg in [MSG_LH_EPS_03, MSG_ESP_05, MSG_TSK_06, MSG_MOTOR_20]:
      self.safety.set_controls_allowed(1)
      if msg == MSG_LH_EPS_03:
        to_push = self._torque_driver_msg(0)
      if msg == MSG_ESP_05:
        to_push = self._user_brake_msg(False)
      if msg == MSG_TSK_06:
        to_push = self._pcm_status_msg(True)
      if msg == MSG_MOTOR_20:
        to_push = self._user_gas_msg(0)
      self.assertTrue(self._rx(to_push))
      to_push[0].data[4] ^= 0xFF
      self.assertFalse(self._rx(to_push))
      self.assertFalse(self.safety.get_controls_allowed())

    # counter
    # reset wrong_counters to zero by sending valid messages
    for i in range(MAX_WRONG_COUNTERS + 1):
      self.__class__.cnt_lh_eps_03 += 1
      self.__class__.cnt_esp_05 += 1
      self.__class__.cnt_tsk_06 += 1
      self.__class__.cnt_motor_20 += 1
      if i < MAX_WRONG_COUNTERS:
        self.safety.set_controls_allowed(1)
        self._rx(self._torque_driver_msg(0))
        self._rx(self._user_brake_msg(False))
        self._rx(self._pcm_status_msg(True))
        self._rx(self._user_gas_msg(0))
      else:
        self.assertFalse(self._rx(self._torque_driver_msg(0)))
        self.assertFalse(self._rx(self._user_brake_msg(False)))
        self.assertFalse(self._rx(self._pcm_status_msg(True)))
        self.assertFalse(self._rx(self._user_gas_msg(0)))
        self.assertFalse(self.safety.get_controls_allowed())

    # restore counters for future tests with a couple of good messages
    for i in range(2):
      self.safety.set_controls_allowed(1)
      self._rx(self._torque_driver_msg(0))
      self._rx(self._user_brake_msg(False))
      self._rx(self._pcm_status_msg(True))
      self._rx(self._user_gas_msg(0))
    self.assertTrue(self.safety.get_controls_allowed())


class TestVolkswagenMqbStockSafety(TestVolkswagenMqbSafety):
  TX_MSGS = [[MSG_HCA_01, 0], [MSG_LDW_02, 0], [MSG_GRA_ACC_01, 0], [MSG_GRA_ACC_01, 2]]
  FWD_BLACKLISTED_ADDRS = {2: [MSG_HCA_01, MSG_LDW_02]}
  FWD_BUS_LOOKUP = {0: 2, 2: 0}

  def setUp(self):
    self.packer = CANPackerPanda("vw_mqb_2010")
    self.safety = libpandasafety_py.libpandasafety
    self.safety.set_safety_hooks(Panda.SAFETY_VOLKSWAGEN_MQB, 0)
    self.safety.init_tests()

  def test_spam_cancel_safety_check(self):
    self.safety.set_controls_allowed(0)
    self.assertTrue(self._tx(self._gra_acc_01_msg(cancel=1)))
    self.assertFalse(self._tx(self._gra_acc_01_msg(resume=1)))
    self.assertFalse(self._tx(self._gra_acc_01_msg(_set=1)))
    # do not block resume if we are engaged already
    self.safety.set_controls_allowed(1)
    self.assertTrue(self._tx(self._gra_acc_01_msg(resume=1)))


if __name__ == "__main__":
  unittest.main()
