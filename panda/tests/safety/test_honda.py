#!/usr/bin/env python3
import unittest
import numpy as np
from panda import Panda
from panda.tests.safety import libpandasafety_py
import panda.tests.safety.common as common
from panda.tests.safety.common import CANPackerPanda, make_msg, \
                                      MAX_WRONG_COUNTERS, UNSAFE_MODE

MAX_BRAKE = 255

class Btn:
  CANCEL = 2
  SET = 3
  RESUME = 4

HONDA_N_HW = 0
HONDA_BG_HW = 1
HONDA_BH_HW = 2


class TestHondaSafety(common.PandaSafetyTest):
  cnt_speed = 0
  cnt_gas = 0
  cnt_button = 0

  PT_BUS = 0

  @classmethod
  def setUpClass(cls):
    if cls.__name__ == "TestHondaSafety":
      cls.packer = None
      cls.safety = None
      raise unittest.SkipTest

  # override these inherited tests. honda doesn't use pcm enable
  def test_disable_control_allowed_from_cruise(self): pass
  def test_enable_control_allowed_from_cruise(self): pass
  def test_cruise_engaged_prev(self): pass

  def _speed_msg(self, speed):
    values = {"XMISSION_SPEED": speed, "COUNTER": self.cnt_speed % 4}
    self.__class__.cnt_speed += 1
    return self.packer.make_can_msg_panda("ENGINE_DATA", self.PT_BUS, values)

  def _button_msg(self, buttons):
    values = {"CRUISE_BUTTONS": buttons, "COUNTER": self.cnt_button % 4}
    self.__class__.cnt_button += 1
    return self.packer.make_can_msg_panda("SCM_BUTTONS", self.PT_BUS, values)

  def _brake_msg(self, brake):
    values = {"BRAKE_PRESSED": brake, "COUNTER": self.cnt_gas % 4}
    self.__class__.cnt_gas += 1
    return self.packer.make_can_msg_panda("POWERTRAIN_DATA", self.PT_BUS, values)

  def _gas_msg(self, gas):
    values = {"PEDAL_GAS": gas, "COUNTER": self.cnt_gas % 4}
    self.__class__.cnt_gas += 1
    return self.packer.make_can_msg_panda("POWERTRAIN_DATA", self.PT_BUS, values)

  def _send_brake_msg(self, brake):
    values = {}
    if self.safety.get_honda_hw() == HONDA_N_HW:
      values = {"COMPUTER_BRAKE": brake}
    return self.packer.make_can_msg_panda("BRAKE_COMMAND", 0, values)

  def _send_steer_msg(self, steer):
    values = {"STEER_TORQUE": steer}
    return self.packer.make_can_msg_panda("STEERING_CONTROL", 0, values)

  def test_resume_button(self):
    self.safety.set_controls_allowed(0)
    self._rx(self._button_msg(Btn.RESUME))
    self.assertTrue(self.safety.get_controls_allowed())

  def test_set_button(self):
    self.safety.set_controls_allowed(0)
    self._rx(self._button_msg(Btn.SET))
    self.assertTrue(self.safety.get_controls_allowed())

  def test_cancel_button(self):
    self.safety.set_controls_allowed(1)
    self._rx(self._button_msg(Btn.CANCEL))
    self.assertFalse(self.safety.get_controls_allowed())

  def test_disengage_on_brake(self):
    self.safety.set_controls_allowed(1)
    self._rx(self._brake_msg(1))
    self.assertFalse(self.safety.get_controls_allowed())

  def test_steer_safety_check(self):
    self.safety.set_controls_allowed(0)
    self.assertTrue(self._tx(self._send_steer_msg(0x0000)))
    self.assertFalse(self._tx(self._send_steer_msg(0x1000)))

  def test_rx_hook(self):

    # TODO: move this test to common
    # checksum checks
    for msg in ["btn", "gas", "speed"]:
      self.safety.set_controls_allowed(1)
      # TODO: add this coverage back by re-running all tests with the acura dbc
      # to_push = self._button_msg(Btn.SET, 0x1A6)  # only in Honda_NIDEC
      if msg == "btn":
        to_push = self._button_msg(Btn.SET)
      if msg == "gas":
        to_push = self._gas_msg(0)
      if msg == "speed":
        to_push = self._speed_msg(0)
      self.assertTrue(self._rx(to_push))
      if msg != "btn":
        to_push[0].RDHR = 0  # invalidate checksum
        self.assertFalse(self._rx(to_push))
        self.assertFalse(self.safety.get_controls_allowed())

    # counter
    # reset wrong_counters to zero by sending valid messages
    for i in range(MAX_WRONG_COUNTERS + 1):
      self.__class__.cnt_speed += 1
      self.__class__.cnt_gas += 1
      self.__class__.cnt_button += 1
      if i < MAX_WRONG_COUNTERS:
        self.safety.set_controls_allowed(1)
        self._rx(self._button_msg(Btn.SET))
        self._rx(self._speed_msg(0))
        self._rx(self._gas_msg(0))
      else:
        self.assertFalse(self._rx(self._button_msg(Btn.SET)))
        self.assertFalse(self._rx(self._speed_msg(0)))
        self.assertFalse(self._rx(self._gas_msg(0)))
        self.assertFalse(self.safety.get_controls_allowed())

    # restore counters for future tests with a couple of good messages
    for i in range(2):
      self.safety.set_controls_allowed(1)
      self._rx(self._button_msg(Btn.SET))
      self._rx(self._speed_msg(0))
      self._rx(self._gas_msg(0))
    self._rx(self._button_msg(Btn.SET))
    self.assertTrue(self.safety.get_controls_allowed())

  def test_tx_hook_on_pedal_pressed(self):
    for mode in [UNSAFE_MODE.DEFAULT, UNSAFE_MODE.DISABLE_DISENGAGE_ON_GAS]:
      for pedal in ['brake', 'gas']:
        self.safety.set_unsafe_mode(mode)
        allow_ctrl = False
        if pedal == 'brake':
          # brake_pressed_prev and vehicle_moving
          self._rx(self._speed_msg(100))
          self._rx(self._brake_msg(1))
        elif pedal == 'gas':
          # gas_pressed_prev
          self._rx(self._gas_msg(1))
          allow_ctrl = mode == UNSAFE_MODE.DISABLE_DISENGAGE_ON_GAS

        self.safety.set_controls_allowed(1)
        hw = self.safety.get_honda_hw()
        if hw == HONDA_N_HW:
          self.safety.set_honda_fwd_brake(False)
          self.assertEqual(allow_ctrl, self._tx(self._send_brake_msg(MAX_BRAKE)))
        self.assertEqual(allow_ctrl, self._tx(self._send_steer_msg(0x1000)))

        # reset status
        self.safety.set_controls_allowed(0)
        self.safety.set_unsafe_mode(UNSAFE_MODE.DEFAULT)
        self._tx(self._send_brake_msg(0))
        self._tx(self._send_steer_msg(0))
        if pedal == 'brake':
          self._rx(self._speed_msg(0))
          self._rx(self._brake_msg(0))
        elif pedal == 'gas':
          self._rx(self._gas_msg(0))


class TestHondaNidecSafety(TestHondaSafety, common.InterceptorSafetyTest):

  TX_MSGS = [[0xE4, 0], [0x194, 0], [0x1FA, 0], [0x200, 0], [0x30C, 0], [0x33D, 0]]
  STANDSTILL_THRESHOLD = 0
  RELAY_MALFUNCTION_ADDR = 0xE4
  RELAY_MALFUNCTION_BUS = 0
  FWD_BLACKLISTED_ADDRS = {2: [0xE4, 0x194, 0x33D, 0x30C]}
  FWD_BUS_LOOKUP = {0: 2, 2: 0}

  INTERCEPTOR_THRESHOLD = 344

  def setUp(self):
    self.packer = CANPackerPanda("honda_civic_touring_2016_can_generated")
    self.safety = libpandasafety_py.libpandasafety
    self.safety.set_safety_hooks(Panda.SAFETY_HONDA_NIDEC, 0)
    self.safety.init_tests_honda()

  # Honda gas gains are the different
  def _interceptor_msg(self, gas, addr):
    to_send = make_msg(0, addr, 6)
    gas2 = gas * 2
    to_send[0].RDLR = ((gas & 0xff) << 8) | ((gas & 0xff00) >> 8) | \
                    ((gas2 & 0xff) << 24) | ((gas2 & 0xff00) << 8)
    return to_send

  def test_fwd_hook(self):
    # normal operation, not forwarding AEB
    self.FWD_BLACKLISTED_ADDRS[2].append(0x1FA)
    self.safety.set_honda_fwd_brake(False)
    super().test_fwd_hook()

    # TODO: test latching until AEB event is over?
    # forwarding AEB brake signal
    self.FWD_BLACKLISTED_ADDRS = {2: [0xE4, 0x194, 0x33D, 0x30C]}
    self.safety.set_honda_fwd_brake(True)
    super().test_fwd_hook()

  def test_brake_safety_check(self):
    for fwd_brake in [False, True]:
      self.safety.set_honda_fwd_brake(fwd_brake)
      for brake in np.arange(0, MAX_BRAKE + 10, 1):
        for controls_allowed in [True, False]:
          self.safety.set_controls_allowed(controls_allowed)
          if fwd_brake:
            send = False  # block openpilot brake msg when fwd'ing stock msg
          elif controls_allowed:
            send = MAX_BRAKE >= brake >= 0
          else:
            send = brake == 0
          self.assertEqual(send, self._tx(self._send_brake_msg(brake)))
    self.safety.set_honda_fwd_brake(False)

  def test_tx_hook_on_interceptor_pressed(self):
    for mode in [UNSAFE_MODE.DEFAULT, UNSAFE_MODE.DISABLE_DISENGAGE_ON_GAS]:
      self.safety.set_unsafe_mode(mode)
      # gas_interceptor_prev > INTERCEPTOR_THRESHOLD
      self._rx(self._interceptor_msg(self.INTERCEPTOR_THRESHOLD+1, 0x201))
      self._rx(self._interceptor_msg(self.INTERCEPTOR_THRESHOLD+1, 0x201))
      allow_ctrl = mode == UNSAFE_MODE.DISABLE_DISENGAGE_ON_GAS

      self.safety.set_controls_allowed(1)
      self.safety.set_honda_fwd_brake(False)
      self.assertEqual(allow_ctrl, self._tx(self._send_brake_msg(MAX_BRAKE)))
      self.assertEqual(allow_ctrl, self._tx(self._interceptor_msg(self.INTERCEPTOR_THRESHOLD, 0x200)))
      self.assertEqual(allow_ctrl, self._tx(self._send_steer_msg(0x1000)))

      # reset status
      self.safety.set_controls_allowed(0)
      self.safety.set_unsafe_mode(UNSAFE_MODE.DEFAULT)
      self._tx(self._send_brake_msg(0))
      self._tx(self._send_steer_msg(0))
      self._tx(self._interceptor_msg(0, 0x200))
      self.safety.set_gas_interceptor_detected(False)


class TestHondaBoschHarnessSafety(TestHondaSafety):
  TX_MSGS = [[0xE4, 0], [0xE5, 0], [0x296, 1], [0x33D, 0]]  # Bosch Harness
  STANDSTILL_THRESHOLD = 0
  RELAY_MALFUNCTION_ADDR = 0xE4
  RELAY_MALFUNCTION_BUS = 0
  FWD_BLACKLISTED_ADDRS = {2: [0xE4, 0xE5, 0x33D]}
  FWD_BUS_LOOKUP = {0: 2, 2: 0}

  PT_BUS = 1

  def setUp(self):
    self.packer = CANPackerPanda("honda_accord_s2t_2018_can_generated")
    self.safety = libpandasafety_py.libpandasafety
    self.safety.set_safety_hooks(Panda.SAFETY_HONDA_BOSCH_HARNESS, 0)
    self.safety.init_tests_honda()

  def _alt_brake_msg(self, brake):
    to_send = make_msg(0, 0x1BE)
    to_send[0].RDLR = 0x10 if brake else 0
    return to_send

  def test_spam_cancel_safety_check(self):
    self.safety.set_controls_allowed(0)
    self.assertTrue(self._tx(self._button_msg(Btn.CANCEL)))
    self.assertFalse(self._tx(self._button_msg(Btn.RESUME)))
    self.assertFalse(self._tx(self._button_msg(Btn.SET)))
    # do not block resume if we are engaged already
    self.safety.set_controls_allowed(1)
    self.assertTrue(self._tx(self._button_msg(Btn.RESUME)))

  def test_alt_disengage_on_brake(self):
    self.safety.set_honda_alt_brake_msg(1)
    self.safety.set_controls_allowed(1)
    self._rx(self._alt_brake_msg(1))
    self.assertFalse(self.safety.get_controls_allowed())

    self.safety.set_honda_alt_brake_msg(0)
    self.safety.set_controls_allowed(1)
    self._rx(self._alt_brake_msg(1))
    self.assertTrue(self.safety.get_controls_allowed())


class TestHondaBoschGiraffeSafety(TestHondaBoschHarnessSafety):
  TX_MSGS = [[0xE4, 2], [0xE5, 2], [0x296, 0], [0x33D, 2]]  # Bosch Giraffe
  STANDSTILL_THRESHOLD = 0
  RELAY_MALFUNCTION_ADDR = 0xE4
  RELAY_MALFUNCTION_BUS = 2
  FWD_BLACKLISTED_ADDRS = {1: [0xE4, 0xE5, 0x33D]}
  FWD_BUS_LOOKUP = {1: 2, 2: 1}

  PT_BUS = 0

  def setUp(self):
    super().setUp()
    self.safety = libpandasafety_py.libpandasafety
    self.safety.set_safety_hooks(Panda.SAFETY_HONDA_BOSCH_GIRAFFE, 0)
    self.safety.init_tests_honda()

  def _send_steer_msg(self, steer):
    values = {"STEER_TORQUE": steer}
    return self.packer.make_can_msg_panda("STEERING_CONTROL", 2, values)


if __name__ == "__main__":
  unittest.main()
