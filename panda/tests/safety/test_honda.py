#!/usr/bin/env python3
import unittest
import numpy as np
from typing import Optional

from panda import Panda
from panda.tests.safety import libpandasafety_py
import panda.tests.safety.common as common
from panda.tests.safety.common import CANPackerPanda, make_msg, MAX_WRONG_COUNTERS, ALTERNATIVE_EXPERIENCE

class Btn:
  NONE = 0
  MAIN = 1
  CANCEL = 2
  SET = 3
  RESUME = 4

HONDA_NIDEC = 0
HONDA_BOSCH = 1


def interceptor_msg(gas, addr):
  to_send = make_msg(0, addr, 6)
  to_send[0].data[0] = (gas & 0xFF00) >> 8
  to_send[0].data[1] = gas & 0xFF
  to_send[0].data[2] = (gas & 0xFF00) >> 8
  to_send[0].data[3] = gas & 0xFF
  return to_send


# Honda safety has several different configurations tested here:
#  * Nidec
#    * normal
#    * alt SCM messages
#    * interceptor
#    * interceptor with alt SCM messages
#  * Bosch
#    * Bosch with Longitudinal Support


class HondaButtonEnableBase(common.PandaSafetyTest):
  # pylint: disable=no-member,abstract-method

  # override these inherited tests since we're using button enable
  def test_disable_control_allowed_from_cruise(self):
    pass

  def test_enable_control_allowed_from_cruise(self):
    pass

  def test_cruise_engaged_prev(self):
    pass

  def test_buttons_with_main_off(self):
    for btn in (Btn.SET, Btn.RESUME, Btn.CANCEL):
      self.safety.set_controls_allowed(1)
      self._rx(self._acc_state_msg(False))
      self._rx(self._button_msg(btn, main_on=False))
      self.assertFalse(self.safety.get_controls_allowed())

  def test_set_resume_buttons(self):
    """
      Both SET and RES should enter controls allowed on their falling edge.
    """
    for btn in (Btn.SET, Btn.RESUME):
      for main_on in (True, False):
        self._rx(self._acc_state_msg(main_on))
        self.safety.set_controls_allowed(0)

        # nothing until falling edge
        for _ in range(10):
          self._rx(self._button_msg(btn, main_on=main_on))
        self.assertFalse(self.safety.get_controls_allowed())

        self._rx(self._button_msg(Btn.NONE, main_on=main_on))
        self.assertEqual(main_on, self.safety.get_controls_allowed(), msg=f"{main_on=} {btn=}")

  def test_main_cancel_buttons(self):
    """
      Both MAIN and CANCEL should exit controls immediately.
    """
    for btn in (Btn.MAIN, Btn.CANCEL):
      self.safety.set_controls_allowed(1)
      self._rx(self._button_msg(btn, main_on=True))
      self.assertFalse(self.safety.get_controls_allowed())

  def test_disengage_on_main(self):
    self.safety.set_controls_allowed(1)
    self._rx(self._acc_state_msg(True))
    self.assertTrue(self.safety.get_controls_allowed())
    self._rx(self._acc_state_msg(False))
    self.assertFalse(self.safety.get_controls_allowed())

  def test_rx_hook(self):

    # TODO: move this test to common
    # checksum checks
    for msg in ["btn", "gas", "speed"]:
      self.safety.set_controls_allowed(1)
      if msg == "btn":
        to_push = self._button_msg(Btn.SET)
      if msg == "gas":
        to_push = self._user_gas_msg(0)
      if msg == "speed":
        to_push = self._speed_msg(0)
      self.assertTrue(self._rx(to_push))
      if msg != "btn":
        to_push[0].data[4] = 0  # invalidate checksum
        to_push[0].data[5] = 0
        to_push[0].data[6] = 0
        to_push[0].data[7] = 0
        self.assertFalse(self._rx(to_push))
        self.assertFalse(self.safety.get_controls_allowed())

    # counter
    # reset wrong_counters to zero by sending valid messages
    for i in range(MAX_WRONG_COUNTERS + 1):
      self.__class__.cnt_speed += 1
      self.__class__.cnt_button += 1
      self.__class__.cnt_powertrain_data += 1
      if i < MAX_WRONG_COUNTERS:
        self.safety.set_controls_allowed(1)
        self._rx(self._button_msg(Btn.SET))
        self._rx(self._speed_msg(0))
        self._rx(self._user_gas_msg(0))
      else:
        self.assertFalse(self._rx(self._button_msg(Btn.SET)))
        self.assertFalse(self._rx(self._speed_msg(0)))
        self.assertFalse(self._rx(self._user_gas_msg(0)))
        self.assertFalse(self.safety.get_controls_allowed())

    # restore counters for future tests with a couple of good messages
    for i in range(2):
      self.safety.set_controls_allowed(1)
      self._rx(self._button_msg(Btn.SET))
      self._rx(self._speed_msg(0))
      self._rx(self._user_gas_msg(0))
    self._rx(self._button_msg(Btn.SET, main_on=True))
    self.assertTrue(self.safety.get_controls_allowed())

  def test_tx_hook_on_pedal_pressed(self):
    for mode in [ALTERNATIVE_EXPERIENCE.DEFAULT, ALTERNATIVE_EXPERIENCE.DISABLE_DISENGAGE_ON_GAS]:
      for pedal in ['brake', 'gas']:
        self.safety.set_alternative_experience(mode)
        allow_ctrl = False
        if pedal == 'brake':
          # brake_pressed_prev and vehicle_moving
          self._rx(self._speed_msg(100))
          self._rx(self._user_brake_msg(1))
        elif pedal == 'gas':
          # gas_pressed_prev
          self._rx(self._user_gas_msg(1))
          allow_ctrl = mode == ALTERNATIVE_EXPERIENCE.DISABLE_DISENGAGE_ON_GAS

        self.safety.set_controls_allowed(1)
        hw = self.safety.get_honda_hw()
        if hw == HONDA_NIDEC:
          self.safety.set_honda_fwd_brake(False)
          self.assertEqual(allow_ctrl, self._tx(self._send_brake_msg(self.MAX_BRAKE)))
        self.assertEqual(allow_ctrl, self._tx(self._send_steer_msg(0x1000)))

        # reset status
        self.safety.set_controls_allowed(0)
        self.safety.set_alternative_experience(ALTERNATIVE_EXPERIENCE.DEFAULT)
        if hw == HONDA_NIDEC:
          self._tx(self._send_brake_msg(0))
        self._tx(self._send_steer_msg(0))
        if pedal == 'brake':
          self._rx(self._speed_msg(0))
          self._rx(self._user_brake_msg(0))
        elif pedal == 'gas':
          self._rx(self._user_gas_msg(0))


class HondaPcmEnableBase(common.PandaSafetyTest):
  # pylint: disable=no-member,abstract-method

  def test_buttons(self):
    """
      Buttons should only cancel in this configuration,
      since our state is tied to the PCM's cruise state.
    """
    for controls_allowed in (True, False):
      for main_on in (True, False):
        # not a valid state
        if controls_allowed and not main_on:
          continue

        for btn in (Btn.SET, Btn.RESUME, Btn.CANCEL):
          self.safety.set_controls_allowed(controls_allowed)
          self._rx(self._acc_state_msg(main_on))

          # btn + none for falling edge
          self._rx(self._button_msg(btn, main_on=main_on))
          self._rx(self._button_msg(Btn.NONE, main_on=main_on))

          if btn == Btn.CANCEL:
            self.assertFalse(self.safety.get_controls_allowed())
          else:
            self.assertEqual(controls_allowed, self.safety.get_controls_allowed())


class HondaBase(common.PandaSafetyTest):
  MAX_BRAKE: float = 255
  PT_BUS: Optional[int] = None  # must be set when inherited
  STEER_BUS: Optional[int] = None  # must be set when inherited
  BUTTONS_BUS: Optional[int] = None  # must be set when inherited, tx on this bus, rx on PT_BUS

  STANDSTILL_THRESHOLD = 0
  RELAY_MALFUNCTION_ADDR = 0xE4
  RELAY_MALFUNCTION_BUS = 0
  FWD_BUS_LOOKUP = {0: 2, 2: 0}

  cnt_speed = 0
  cnt_button = 0
  cnt_brake = 0
  cnt_powertrain_data = 0
  cnt_acc_state = 0

  @classmethod
  def setUpClass(cls):
    if cls.__name__ == "HondaBase":
      cls.packer = None
      cls.safety = None
      raise unittest.SkipTest

  def _powertrain_data_msg(self, cruise_on=None, brake_pressed=None, gas_pressed=None):
    # preserve the state
    if cruise_on is None:
      # or'd with controls allowed since the tests use it to "enable" cruise
      cruise_on = self.safety.get_cruise_engaged_prev() or self.safety.get_controls_allowed()
    if brake_pressed is None:
      brake_pressed = self.safety.get_brake_pressed_prev()
    if gas_pressed is None:
      gas_pressed = self.safety.get_gas_pressed_prev()

    values = {
      "ACC_STATUS": cruise_on,
      "BRAKE_PRESSED": brake_pressed,
      "PEDAL_GAS": gas_pressed,
      "COUNTER": self.cnt_powertrain_data % 4
    }
    self.__class__.cnt_powertrain_data += 1
    return self.packer.make_can_msg_panda("POWERTRAIN_DATA", self.PT_BUS, values)

  def _pcm_status_msg(self, enable):
    return self._powertrain_data_msg(cruise_on=enable)

  def _speed_msg(self, speed):
    values = {"XMISSION_SPEED": speed, "COUNTER": self.cnt_speed % 4}
    self.__class__.cnt_speed += 1
    return self.packer.make_can_msg_panda("ENGINE_DATA", self.PT_BUS, values)

  def _acc_state_msg(self, main_on):
    values = {"MAIN_ON": main_on, "COUNTER": self.cnt_acc_state % 4}
    self.__class__.cnt_acc_state += 1
    return self.packer.make_can_msg_panda("SCM_FEEDBACK", self.PT_BUS, values)

  def _button_msg(self, buttons, main_on=False, bus=None):
    bus = self.PT_BUS if bus is None else bus
    values = {"CRUISE_BUTTONS": buttons, "COUNTER": self.cnt_button % 4}
    self.__class__.cnt_button += 1
    return self.packer.make_can_msg_panda("SCM_BUTTONS", bus, values)

  def _user_brake_msg(self, brake):
    return self._powertrain_data_msg(brake_pressed=brake)

  def _user_gas_msg(self, gas):
    return self._powertrain_data_msg(gas_pressed=gas)

  def _send_steer_msg(self, steer):
    values = {"STEER_TORQUE": steer}
    return self.packer.make_can_msg_panda("STEERING_CONTROL", self.STEER_BUS, values)

  def _send_brake_msg(self, brake):
    # must be implemented when inherited
    raise NotImplementedError

  def test_disengage_on_brake(self):
    self.safety.set_controls_allowed(1)
    self._rx(self._user_brake_msg(1))
    self.assertFalse(self.safety.get_controls_allowed())

  def test_steer_safety_check(self):
    self.safety.set_controls_allowed(0)
    self.assertTrue(self._tx(self._send_steer_msg(0x0000)))
    self.assertFalse(self._tx(self._send_steer_msg(0x1000)))


# ********************* Honda Nidec **********************


class TestHondaNidecSafetyBase(HondaBase):
  TX_MSGS = [[0xE4, 0], [0x194, 0], [0x1FA, 0], [0x200, 0], [0x30C, 0], [0x33D, 0]]
  FWD_BLACKLISTED_ADDRS = {2: [0xE4, 0x194, 0x33D, 0x30C]}

  PT_BUS = 0
  STEER_BUS = 0
  BUTTONS_BUS = 0

  INTERCEPTOR_THRESHOLD = 492

  @classmethod
  def setUpClass(cls):
    if cls.__name__ == "TestHondaNidecSafetyBase":
      cls.safety = None
      raise unittest.SkipTest

  def setUp(self):
    self.packer = CANPackerPanda("honda_civic_touring_2016_can_generated")
    self.safety = libpandasafety_py.libpandasafety
    self.safety.set_safety_hooks(Panda.SAFETY_HONDA_NIDEC, 0)
    self.safety.init_tests_honda()

  def _interceptor_gas_cmd(self, gas):
    return interceptor_msg(gas, 0x200)

  def _interceptor_user_gas(self, gas):
    return interceptor_msg(gas, 0x201)

  def _send_brake_msg(self, brake):
    values = {"COMPUTER_BRAKE": brake}
    return self.packer.make_can_msg_panda("BRAKE_COMMAND", 0, values)

  def _send_acc_hud_msg(self, pcm_gas, pcm_speed):
    # Used to control ACC on Nidec without pedal
    values = {"PCM_GAS": pcm_gas, "PCM_SPEED": pcm_speed}
    return self.packer.make_can_msg_panda("ACC_HUD", 0, values)

  def test_acc_hud_safety_check(self):
    for controls_allowed in [True, False]:
      self.safety.set_controls_allowed(controls_allowed)
      for pcm_gas in range(0, 0xc6):
        for pcm_speed in range(0, 100):
          send = True if controls_allowed else pcm_gas == 0 and pcm_speed == 0
          self.assertEqual(send, self.safety.safety_tx_hook(self._send_acc_hud_msg(pcm_gas, pcm_speed)))

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
      for brake in np.arange(0, self.MAX_BRAKE + 10, 1):
        for controls_allowed in [True, False]:
          self.safety.set_controls_allowed(controls_allowed)
          if fwd_brake:
            send = False  # block openpilot brake msg when fwd'ing stock msg
          elif controls_allowed:
            send = self.MAX_BRAKE >= brake >= 0
          else:
            send = brake == 0
          self.assertEqual(send, self._tx(self._send_brake_msg(brake)))
    self.safety.set_honda_fwd_brake(False)

  def test_tx_hook_on_interceptor_pressed(self):
    for mode in [ALTERNATIVE_EXPERIENCE.DEFAULT, ALTERNATIVE_EXPERIENCE.DISABLE_DISENGAGE_ON_GAS]:
      self.safety.set_alternative_experience(mode)
      # gas_interceptor_prev > INTERCEPTOR_THRESHOLD
      self._rx(self._interceptor_user_gas(self.INTERCEPTOR_THRESHOLD + 1))
      self._rx(self._interceptor_user_gas(self.INTERCEPTOR_THRESHOLD + 1))
      allow_ctrl = mode == ALTERNATIVE_EXPERIENCE.DISABLE_DISENGAGE_ON_GAS

      self.safety.set_controls_allowed(1)
      self.safety.set_honda_fwd_brake(False)

      # Test we allow lateral, but never longitudinal
      self.assertFalse(self._tx(self._interceptor_gas_cmd(self.INTERCEPTOR_THRESHOLD)))
      self.assertFalse(self._tx(self._send_brake_msg(self.MAX_BRAKE)))
      self.assertEqual(allow_ctrl, self._tx(self._send_steer_msg(0x1000)))


class TestHondaNidecSafety(HondaPcmEnableBase, TestHondaNidecSafetyBase):
  """
    Covers the Honda Nidec safety mode
  """

  # Nidec doesn't disengage on falling edge of cruise. See comment in safety_honda.h
  def test_disable_control_allowed_from_cruise(self):
    pass


class TestHondaNidecInterceptorSafety(TestHondaNidecSafety, common.InterceptorSafetyTest):
  """
    Covers the Honda Nidec safety mode with a gas interceptor
  """


class TestHondaNidecAltSafety(TestHondaNidecSafety):
  """
    Covers the Honda Nidec safety mode with alt SCM messages
  """
  def setUp(self):
    self.packer = CANPackerPanda("acura_ilx_2016_can_generated")
    self.safety = libpandasafety_py.libpandasafety
    self.safety.set_safety_hooks(Panda.SAFETY_HONDA_NIDEC, Panda.FLAG_HONDA_NIDEC_ALT)
    self.safety.init_tests_honda()

  def _acc_state_msg(self, main_on):
    values = {"MAIN_ON": main_on, "COUNTER": self.cnt_acc_state % 4}
    self.__class__.cnt_acc_state += 1
    return self.packer.make_can_msg_panda("SCM_BUTTONS", self.PT_BUS, values)

  def _button_msg(self, buttons, main_on=False, bus=None):
    bus = self.PT_BUS if bus is None else bus
    values = {"CRUISE_BUTTONS": buttons, "MAIN_ON": main_on, "COUNTER": self.cnt_button % 4}
    self.__class__.cnt_button += 1
    return self.packer.make_can_msg_panda("SCM_BUTTONS", bus, values)


class TestHondaNidecAltInterceptorSafety(TestHondaNidecSafety, common.InterceptorSafetyTest):
  """
    Covers the Honda Nidec safety mode with alt SCM messages and gas interceptor
  """
  def setUp(self):
    self.packer = CANPackerPanda("acura_ilx_2016_can_generated")
    self.safety = libpandasafety_py.libpandasafety
    self.safety.set_safety_hooks(Panda.SAFETY_HONDA_NIDEC, Panda.FLAG_HONDA_NIDEC_ALT)
    self.safety.init_tests_honda()

  def _acc_state_msg(self, main_on):
    values = {"MAIN_ON": main_on, "COUNTER": self.cnt_acc_state % 4}
    self.__class__.cnt_acc_state += 1
    return self.packer.make_can_msg_panda("SCM_BUTTONS", self.PT_BUS, values)

  def _button_msg(self, buttons, main_on=False, bus=None):
    bus = self.PT_BUS if bus is None else bus
    values = {"CRUISE_BUTTONS": buttons, "MAIN_ON": main_on, "COUNTER": self.cnt_button % 4}
    self.__class__.cnt_button += 1
    return self.packer.make_can_msg_panda("SCM_BUTTONS", bus, values)



# ********************* Honda Bosch **********************


class TestHondaBoschSafetyBase(HondaBase):
  PT_BUS = 1
  STEER_BUS = 0
  BUTTONS_BUS = 1

  TX_MSGS = [[0xE4, 0], [0xE5, 0], [0x296, 1], [0x33D, 0], [0x33DA, 0], [0x33DB, 0]]
  FWD_BLACKLISTED_ADDRS = {2: [0xE4, 0xE5, 0x33D, 0x33DA, 0x33DB]}

  @classmethod
  def setUpClass(cls):
    if cls.__name__ == "TestHondaBoschSafetyBase":
      cls.packer = None
      cls.safety = None
      raise unittest.SkipTest

  def setUp(self):
    self.packer = CANPackerPanda("honda_accord_2018_can_generated")
    self.safety = libpandasafety_py.libpandasafety

  def _alt_brake_msg(self, brake):
    values = {"BRAKE_PRESSED": brake, "COUNTER": self.cnt_brake % 4}
    self.__class__.cnt_brake += 1
    return self.packer.make_can_msg_panda("BRAKE_MODULE", self.PT_BUS, values)

  def _send_brake_msg(self, brake):
    pass

  # TODO: add back in once alternative brake checksum/counter validation is added
  # def test_alt_brake_rx_hook(self):
  #   self.safety.set_honda_alt_brake_msg(1)
  #   self.safety.set_controls_allowed(1)
  #   to_push = self._alt_brake_msg(0)
  #   self.assertTrue(self._rx(to_push))
  #   to_push[0].RDLR = to_push[0].RDLR & 0xFFF0FFFF # invalidate checksum
  #   self.assertFalse(self._rx(to_push))
  #   self.assertFalse(self.safety.get_controls_allowed())
  def test_alt_disengage_on_brake(self):
    self.safety.set_honda_alt_brake_msg(1)
    self.safety.set_controls_allowed(1)
    self._rx(self._alt_brake_msg(1))
    self.assertFalse(self.safety.get_controls_allowed())

    self.safety.set_honda_alt_brake_msg(0)
    self.safety.set_controls_allowed(1)
    self._rx(self._alt_brake_msg(1))
    self.assertTrue(self.safety.get_controls_allowed())

  def test_spam_cancel_safety_check(self):
    self.safety.set_controls_allowed(0)
    self.assertTrue(self._tx(self._button_msg(Btn.CANCEL, bus=self.BUTTONS_BUS)))
    self.assertFalse(self._tx(self._button_msg(Btn.RESUME, bus=self.BUTTONS_BUS)))
    self.assertFalse(self._tx(self._button_msg(Btn.SET, bus=self.BUTTONS_BUS)))
    # do not block resume if we are engaged already
    self.safety.set_controls_allowed(1)
    self.assertTrue(self._tx(self._button_msg(Btn.RESUME, bus=self.BUTTONS_BUS)))


class TestHondaBoschSafety(HondaPcmEnableBase, TestHondaBoschSafetyBase):
  """
    Covers the Honda Bosch safety mode with stock longitudinal
  """
  def setUp(self):
    super().setUp()
    self.safety.set_safety_hooks(Panda.SAFETY_HONDA_BOSCH, 0)
    self.safety.init_tests_honda()


class TestHondaBoschLongSafety(HondaButtonEnableBase, TestHondaBoschSafetyBase):
  """
    Covers the Honda Bosch safety mode with longitudinal control
  """
  NO_GAS = -30000
  MAX_GAS = 2000
  MAX_BRAKE = -3.5

  STEER_BUS = 1
  TX_MSGS = [[0xE4, 1], [0x1DF, 1], [0x1EF, 1], [0x1FA, 1], [0x30C, 1], [0x33D, 1], [0x33DA, 1], [0x33DB, 1], [0x39F, 1], [0x18DAB0F1, 1]]
  FWD_BLACKLISTED_ADDRS = {2: [0xE4, 0xE5, 0x33D, 0x33DA, 0x33DB]}

  def setUp(self):
    super().setUp()
    self.safety.set_safety_hooks(Panda.SAFETY_HONDA_BOSCH, Panda.FLAG_HONDA_BOSCH_LONG)
    self.safety.init_tests_honda()

  def _send_gas_brake_msg(self, gas, accel):
    values = {
      "GAS_COMMAND": gas,
      "ACCEL_COMMAND": accel,
      "BRAKE_REQUEST": accel < 0,
    }
    return self.packer.make_can_msg_panda("ACC_CONTROL", self.PT_BUS, values)

  # Longitudinal doesn't need to send buttons
  def test_spam_cancel_safety_check(self):
    pass

  def test_diagnostics(self):
    tester_present = common.package_can_msg((0x18DAB0F1, 0, b"\x02\x3E\x80\x00\x00\x00\x00\x00", self.PT_BUS))
    self.assertTrue(self.safety.safety_tx_hook(tester_present))

    not_tester_present = common.package_can_msg((0x18DAB0F1, 0, b"\x03\xAA\xAA\x00\x00\x00\x00\x00", self.PT_BUS))
    self.assertFalse(self.safety.safety_tx_hook(not_tester_present))

  def test_radar_alive(self):
    # If the radar knockout failed, make sure the relay malfunction is shown
    self.assertFalse(self.safety.get_relay_malfunction())
    self._rx(make_msg(self.PT_BUS, 0x1DF, 8))
    self.assertTrue(self.safety.get_relay_malfunction())

  def test_gas_safety_check(self):
    for controls_allowed in [True, False]:
      for gas in np.arange(self.NO_GAS, self.MAX_GAS + 2000, 100):
        accel = 0 if gas < 0 else gas / 1000
        self.safety.set_controls_allowed(controls_allowed)
        send = gas <= self.MAX_GAS if controls_allowed else gas == self.NO_GAS
        self.assertEqual(send, self.safety.safety_tx_hook(self._send_gas_brake_msg(gas, accel)), gas)

  def test_brake_safety_check(self):
    for controls_allowed in [True, False]:
      for accel in np.arange(0, self.MAX_BRAKE - 1, -0.01):
        accel = round(accel, 2) # floats might not hit exact boundary conditions without rounding
        self.safety.set_controls_allowed(controls_allowed)
        send = self.MAX_BRAKE <= accel <= 0 if controls_allowed else accel == 0
        self.assertEqual(send, self._tx(self._send_gas_brake_msg(self.NO_GAS, accel)), (controls_allowed, accel))


class TestHondaBoschRadarless(HondaPcmEnableBase, TestHondaBoschSafetyBase):
  PT_BUS = 0
  STEER_BUS = 0
  BUTTONS_BUS = 2  # camera controls ACC, need to send buttons on bus 2

  TX_MSGS = [[0xE4, 0], [0x296, 2], [0x33D, 0]]
  FWD_BLACKLISTED_ADDRS = {2: [0xE4, 0xE5, 0x33D, 0x33DA, 0x33DB]}

  def setUp(self):
    self.packer = CANPackerPanda("honda_civic_ex_2022_can_generated")
    self.safety = libpandasafety_py.libpandasafety
    self.safety.set_safety_hooks(Panda.SAFETY_HONDA_BOSCH, Panda.FLAG_HONDA_RADARLESS)
    self.safety.init_tests_honda()

  def test_alt_disengage_on_brake(self):
    # These cars do not have 0x1BE (BRAKE_MODULE)
    pass


if __name__ == "__main__":
  unittest.main()
