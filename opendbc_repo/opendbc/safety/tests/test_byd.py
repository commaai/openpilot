#!/usr/bin/env python3
import itertools
import unittest
import numpy as np

from opendbc.car.byd.carcontroller import get_safety_CP
from opendbc.car.byd.values import CarControllerParams
from opendbc.car.lateral import get_max_angle_delta_vm, get_max_angle_vm
from opendbc.car.structs import CarParams
from opendbc.car.vehicle_model import VehicleModel
from opendbc.safety.tests.libsafety import libsafety_py
import opendbc.safety.tests.common as common
from opendbc.safety.tests.common import CANPackerSafety, away_round

STEERING_MODULE_ADAS = 0x1E2
LKAS_HUD_ADAS = 0x316
PCM_BUTTONS = 0x3B0


def safety_max_can(max_angle_float, can_offset=0):
  # Matches C: max_angle_can = (int)(max_angle * 10 + 1.) which is floor(max_angle * 10) + 1
  return int(max_angle_float * 10 + 1.) + can_offset


class TestBydSafety(common.CarSafetyTest, common.AngleSteeringSafetyTest):
  RELAY_MALFUNCTION_ADDRS = {0: (STEERING_MODULE_ADAS, LKAS_HUD_ADAS)}
  FWD_BLACKLISTED_ADDRS = {2: [STEERING_MODULE_ADAS, LKAS_HUD_ADAS]}
  TX_MSGS = [[STEERING_MODULE_ADAS, 0], [LKAS_HUD_ADAS, 0], [PCM_BUTTONS, 0]]

  MAIN_BUS = 0
  CAM_BUS = 2

  STEER_ANGLE_MAX = 390  # deg, EPS fault limit
  DEG_TO_CAN = 10

  # BYD uses get_max_angle_delta_vm and get_max_angle_vm for lateral accel and jerk limits
  ANGLE_RATE_BP = None
  ANGLE_RATE_UP = None
  ANGLE_RATE_DOWN = None

  # Real time limits
  LATERAL_FREQUENCY = 50  # Hz

  def _get_steer_cmd_angle_max(self, speed):
    return get_max_angle_vm(max(speed, 1), self.VM, CarControllerParams)

  def setUp(self):
    self.VM = VehicleModel(get_safety_CP())
    self.packer = CANPackerSafety("byd_atto3")
    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(CarParams.SafetyModel.byd, 0)
    self.safety.init_tests()

  def _angle_cmd_msg(self, angle: float, enabled: bool, increment_timer: bool = True):
    values = {"STEER_ANGLE": angle, "STEER_REQ": 1 if enabled else 0, "STEER_REQ_ACTIVE_LOW": 0 if enabled else 1}
    if increment_timer:
      self.safety.set_timer(self.__class__.cnt_angle_cmd * int(1e6 / self.LATERAL_FREQUENCY))
      self.__class__.cnt_angle_cmd += 1
    return self.packer.make_can_msg_safety("STEERING_MODULE_ADAS", self.MAIN_BUS, values)

  cnt_angle_cmd = 0

  def _angle_meas_msg(self, angle: float):
    values = {"STEER_ANGLE_2": angle}
    return self.packer.make_can_msg_safety("STEER_MODULE_2", self.MAIN_BUS, values)

  def _pcm_status_msg(self, enable):
    # ACC_STATE: 0=OFF, 3=ACC_ACTIVE
    values = {"ACC_STATE": 3 if enable else 0}
    return self.packer.make_can_msg_safety("ACC_HUD_ADAS", self.CAM_BUS, values)

  def _speed_msg(self, speed):
    values = {"WHEELSPEED_CLEAN": speed * 3.6}
    return self.packer.make_can_msg_safety("WHEELSPEED_CLEAN", self.MAIN_BUS, values)

  def _user_brake_msg(self, brake):
    values = {"BRAKE_PRESSED": 1 if brake else 0}
    return self.packer.make_can_msg_safety("DRIVE_STATE", self.MAIN_BUS, values)

  def _user_gas_msg(self, gas):
    values = {"RAW_THROTTLE": int(gas * 100)}
    return self.packer.make_can_msg_safety("DRIVE_STATE", self.MAIN_BUS, values)

  def test_cruise_buttons(self):
    buttons = ("SET_BTN", "RES_BTN", "LKAS_ON_BTN", "DEC_DISTANCE_BTN", "INC_DISTANCE_BTN", "ACC_ON_BTN")
    for cruise_engaged, controls_allowed in itertools.product((False, True), repeat=2):
      self.assertTrue(self._rx(self._pcm_status_msg(cruise_engaged)))
      self.safety.set_controls_allowed(controls_allowed)
      for pressed in itertools.product((False, True), repeat=len(buttons)):
        values = dict(zip(buttons, pressed, strict=True))
        values.update(SET_ME_1_1=1, SET_ME_1_2=1)
        with self.subTest(cruise_engaged=cruise_engaged, controls_allowed=controls_allowed, buttons=pressed):
          msg = self.packer.make_can_msg_safety("PCM_BUTTONS", self.MAIN_BUS, values)
          should_tx = not any(pressed[:-1]) and (not pressed[-1] or cruise_engaged)
          self.assertEqual(should_tx, self._tx(msg))

  def test_rx_checksums(self):
    for name, bus, signal, initial, corrupt in (
      ("WHEELSPEED_CLEAN", self.MAIN_BUS, "WHEELSPEED_CLEAN", 72, 0),
      ("ACC_HUD_ADAS", self.CAM_BUS, "ACC_STATE", 0, 3),
    ):
      for byte in range(8):
        with self.subTest(message=name, byte=byte):
          self.safety.set_safety_hooks(CarParams.SafetyModel.byd, 0)
          self.safety.init_tests()
          for counter in range(1, 17):
            msg = self.packer.make_can_msg_safety(name, bus, {signal: initial, "COUNTER": counter % 16})
            self.assertTrue(self._rx(msg))
          speed_min = self.safety.get_vehicle_speed_min()
          speed_max = self.safety.get_vehicle_speed_max()
          msg = self.packer.make_can_msg_safety(name, bus, {signal: corrupt, "COUNTER": 1})
          msg[0].data[byte] ^= 0xFF
          self.safety.set_controls_allowed(name == "WHEELSPEED_CLEAN")
          self.assertFalse(self._rx(msg))
          self.assertFalse(self.safety.get_controls_allowed())
          self.assertEqual(speed_min, self.safety.get_vehicle_speed_min())
          self.assertEqual(speed_max, self.safety.get_vehicle_speed_max())

  def test_rx_counters(self):
    for name, bus, signal, value in (
      ("WHEELSPEED_CLEAN", self.MAIN_BUS, "WHEELSPEED_CLEAN", 72),
      ("ACC_HUD_ADAS", self.CAM_BUS, "ACC_STATE", 3),
    ):
      with self.subTest(message=name):
        # Check both counter locations and rollover with valid checksums.
        for counter in range(1, 33):
          msg = self.packer.make_can_msg_safety(name, bus, {signal: value, "COUNTER": counter % 16})
          self.assertTrue(self._rx(msg))
        self.safety.set_controls_allowed(True)
        # Replayed frames are rejected after the common counter tolerance.
        for i in range(common.MAX_WRONG_COUNTERS + 1):
          should_rx = i < common.MAX_WRONG_COUNTERS - 1
          self.assertEqual(should_rx, self._rx(msg))
          self.assertEqual(should_rx, self.safety.get_controls_allowed())
        # A valid sequence clears the counter faults.
        for counter in range(1, common.MAX_WRONG_COUNTERS + 1):
          msg = self.packer.make_can_msg_safety(name, bus, {signal: 0, "COUNTER": counter})
          self.assertTrue(self._rx(msg))

  def test_angle_cmd_when_enabled(self):
    # We properly test lateral acceleration and jerk below
    pass

  def test_lateral_accel_limit(self):
    for speed in np.linspace(0, 40, 100):
      speed = max(speed, 1)
      # match both CAN encoding (0.1 kph/LSB) and VEHICLE_SPEED_FACTOR=1000 rounding in UPDATE_VEHICLE_SPEED
      sent = speed + 1
      sent_can = away_round(sent / 0.1 * 3.6) * 0.1 / 3.6
      speed = round(sent_can * 1000) / 1000 - 1
      for sign in (-1, 1):
        self.safety.set_controls_allowed(True)
        self._reset_speed_measurement(speed + 1)  # safety fudges the speed

        max_angle_float = get_max_angle_vm(speed, self.VM, CarControllerParams)

        # at limit (safety tolerance adds 1 CAN unit)
        max_angle_can = safety_max_can(max_angle_float)
        max_angle_can = min(max_angle_can, self.STEER_ANGLE_MAX * self.DEG_TO_CAN)
        max_angle = sign * max_angle_can / self.DEG_TO_CAN
        self.safety.set_desired_angle_last(sign * max_angle_can)

        self.assertTrue(self._tx(self._angle_cmd_msg(max_angle, True)))

        # 1 unit above limit
        over_can = safety_max_can(max_angle_float, 1)
        over_can_clipped = min(over_can, self.STEER_ANGLE_MAX * self.DEG_TO_CAN)
        over_angle = sign * over_can_clipped / self.DEG_TO_CAN
        self._tx(self._angle_cmd_msg(over_angle, True))

        # at low speeds max angle is above STEER_ANGLE_MAX, so adding 1 has no effect
        should_tx = over_can >= self.STEER_ANGLE_MAX * self.DEG_TO_CAN
        self.assertEqual(should_tx, self._tx(self._angle_cmd_msg(over_angle, True)))

  def test_lateral_jerk_limit(self):
    for speed in np.linspace(0, 40, 100):
      speed = max(speed, 1)
      # match both CAN encoding (0.1 kph/LSB) and VEHICLE_SPEED_FACTOR=1000 rounding in UPDATE_VEHICLE_SPEED
      sent = speed + 1
      sent_can = away_round(sent / 0.1 * 3.6) * 0.1 / 3.6
      speed = round(sent_can * 1000) / 1000 - 1
      for sign in (-1, 1):
        self.safety.set_controls_allowed(True)
        self._reset_speed_measurement(speed + 1)  # safety fudges the speed
        self._tx(self._angle_cmd_msg(0, True))

        max_delta_float = get_max_angle_delta_vm(speed, self.VM, CarControllerParams)

        # Stay within limits
        # Up
        max_delta_can = safety_max_can(max_delta_float)
        max_angle_delta = sign * max_delta_can / self.DEG_TO_CAN
        self.assertTrue(self._tx(self._angle_cmd_msg(max_angle_delta, True)))

        # Don't change
        self.assertTrue(self._tx(self._angle_cmd_msg(max_angle_delta, True)))

        # Down
        self.assertTrue(self._tx(self._angle_cmd_msg(0, True)))

        # Inject too high rates
        # Up
        over_delta_can = safety_max_can(max_delta_float, 1)
        max_angle_delta = sign * over_delta_can / self.DEG_TO_CAN
        self.assertFalse(self._tx(self._angle_cmd_msg(max_angle_delta, True)))

        # Don't change
        self.safety.set_desired_angle_last(sign * over_delta_can)
        self.assertTrue(self._tx(self._angle_cmd_msg(max_angle_delta, True)))

        # Down
        self.assertFalse(self._tx(self._angle_cmd_msg(0, True)))

        # Recover
        self.assertTrue(self._tx(self._angle_cmd_msg(0, True)))


if __name__ == "__main__":
  unittest.main()
