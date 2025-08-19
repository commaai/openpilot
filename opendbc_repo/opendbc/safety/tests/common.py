import os
import abc
import math
import unittest
import importlib
import numpy as np
from collections.abc import Callable

from opendbc.can import CANPacker
from opendbc.safety import ALTERNATIVE_EXPERIENCE
from opendbc.safety.tests.libsafety import libsafety_py

MAX_WRONG_COUNTERS = 5
MAX_SAMPLE_VALS = 6
VEHICLE_SPEED_FACTOR = 1000
RT_INTERVAL = 250000  # 250ms

# Max allowed delta between car speeds
MAX_SPEED_DELTA = 2.0  # m/s

MessageFunction = Callable[[float], libsafety_py.CANPacket]


def sign_of(a):
  return 1 if a > 0 else -1


def away_round(x):
  # non-banker's/away from zero rounding, C++ CANParser uses this style
  return math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)


def round_speed(v):
  return round(v * VEHICLE_SPEED_FACTOR) / VEHICLE_SPEED_FACTOR


def make_msg(bus, addr, length=8, dat=None):
  if dat is None:
    dat = b'\x00' * length
  return libsafety_py.make_CANPacket(addr, bus, dat)


class CANPackerPanda(CANPacker):
  def make_can_msg_panda(self, name_or_addr, bus, values, fix_checksum=None):
    msg = self.make_can_msg(name_or_addr, bus, values)
    if fix_checksum is not None:
      msg = fix_checksum(msg)
    addr, dat, bus = msg
    return libsafety_py.make_CANPacket(addr, bus, dat)


def add_regen_tests(cls):
  """Dynamically adds regen tests for all user brake tests."""

  # only rx/user brake tests, not brake command
  found_tests = [func for func in dir(cls) if func.startswith("test_") and "user_brake" in func]
  assert len(found_tests) >= 3, "Failed to detect known brake tests"

  for test in found_tests:
    def _make_regen_test(brake_func):
      def _regen_test(self):
        # only for safety modes with a regen message
        if self._user_regen_msg(0) is None:
          raise unittest.SkipTest("Safety mode implements no _user_regen_msg")

        getattr(self, brake_func)(self._user_regen_msg, self.safety.get_regen_braking_prev)
      return _regen_test

    setattr(cls, test.replace("brake", "regen"), _make_regen_test(test))

  return cls


class PandaSafetyTestBase(unittest.TestCase):
  safety: libsafety_py.Panda

  @classmethod
  def setUpClass(cls):
    if cls.__name__ == "PandaSafetyTestBase":
      cls.safety = None
      raise unittest.SkipTest

  def _reset_safety_hooks(self):
    self.safety.set_safety_hooks(self.safety.get_current_safety_mode(),
                                 self.safety.get_current_safety_param())

  def _rx(self, msg):
    return self.safety.safety_rx_hook(msg)

  def _tx(self, msg):
    return self.safety.safety_tx_hook(msg)

  def _generic_limit_safety_check(self, msg_function: MessageFunction, min_allowed_value: float, max_allowed_value: float,
                                  min_possible_value: float, max_possible_value: float, test_delta: float = 1, inactive_value: float = 0,
                                  msg_allowed = True, additional_setup: Callable[[float], None] | None = None):
    """
      Enforces that a signal within a message is only allowed to be sent within a specific range, min_allowed_value -> max_allowed_value.
      Tests the range of min_possible_value -> max_possible_value with a delta of test_delta.
      Message is also only allowed to be sent when controls_allowed is true, unless the value is equal to inactive_value.
      Message is never allowed if msg_allowed is false, for example when stock longitudinal is enabled and you are sending acceleration requests.
      additional_setup is used for extra setup before each _tx, ex: for setting the previous torque for rate limits
    """

    # Ensure that we at least test the allowed_value range
    self.assertGreater(max_possible_value, max_allowed_value)
    self.assertLessEqual(min_possible_value, min_allowed_value)

    for controls_allowed in [False, True]:
      # enforce we don't skip over 0 or inactive
      for v in np.concatenate((np.arange(min_possible_value, max_possible_value, test_delta), np.array([0, inactive_value]))):
        v = round(v, 2)  # floats might not hit exact boundary conditions without rounding
        self.safety.set_controls_allowed(controls_allowed)
        if additional_setup is not None:
          additional_setup(v)
        should_tx = controls_allowed and min_allowed_value <= v <= max_allowed_value
        should_tx = (should_tx or v == inactive_value) and msg_allowed
        self.assertEqual(self._tx(msg_function(v)), should_tx, (controls_allowed, should_tx, v))

  def _common_measurement_test(self, msg_func: Callable, min_value: float, max_value: float, factor: float,
                               meas_min_func: Callable[[], int], meas_max_func: Callable[[], int]):
    """Tests accurate measurement parsing, and that the struct is reset on safety mode init"""
    for val in np.arange(min_value, max_value, 0.5):
      for i in range(MAX_SAMPLE_VALS):
        self.assertTrue(self._rx(msg_func(val + i * 0.1)))

      # assert close by one decimal place
      self.assertAlmostEqual(meas_min_func() / factor, val, delta=0.1)
      self.assertAlmostEqual(meas_max_func() / factor - 0.5, val, delta=0.1)

      # ensure sample_t is reset on safety init
      self._reset_safety_hooks()
      self.assertEqual(meas_min_func(), 0)
      self.assertEqual(meas_max_func(), 0)


class LongitudinalAccelSafetyTest(PandaSafetyTestBase, abc.ABC):

  LONGITUDINAL = True
  MAX_ACCEL: float = 2.0
  MIN_ACCEL: float = -3.5
  INACTIVE_ACCEL: float = 0.0

  @classmethod
  def setUpClass(cls):
    if cls.__name__ == "LongitudinalAccelSafetyTest":
      cls.safety = None
      raise unittest.SkipTest

  @abc.abstractmethod
  def _accel_msg(self, accel: float):
    pass

  def test_accel_limits_correct(self):
    self.assertGreater(self.MAX_ACCEL, 0)
    self.assertLess(self.MIN_ACCEL, 0)

  def test_accel_actuation_limits(self):
    limits = ((self.MIN_ACCEL, self.MAX_ACCEL, ALTERNATIVE_EXPERIENCE.DEFAULT),
              (self.MIN_ACCEL, self.MAX_ACCEL, ALTERNATIVE_EXPERIENCE.RAISE_LONGITUDINAL_LIMITS_TO_ISO_MAX))

    for min_accel, max_accel, alternative_experience in limits:
      # enforce we don't skip over 0 or inactive accel
      for accel in np.concatenate((np.arange(min_accel - 1, max_accel + 1, 0.05), [0, self.INACTIVE_ACCEL])):
        accel = round(accel, 2)  # floats might not hit exact boundary conditions without rounding
        for controls_allowed in [True, False]:
          self.safety.set_controls_allowed(controls_allowed)
          self.safety.set_alternative_experience(alternative_experience)
          if self.LONGITUDINAL:
            should_tx = controls_allowed and min_accel <= accel <= max_accel
            should_tx = should_tx or accel == self.INACTIVE_ACCEL
          else:
            should_tx = False
          self.assertEqual(should_tx, self._tx(self._accel_msg(accel)))


class LongitudinalGasBrakeSafetyTest(PandaSafetyTestBase, abc.ABC):

  MIN_BRAKE: int = 0
  MAX_BRAKE: int | None = None
  MAX_POSSIBLE_BRAKE: int | None = None

  MIN_GAS: int = 0
  MAX_GAS: int | None = None
  INACTIVE_GAS = 0
  MIN_POSSIBLE_GAS: int = 0.
  MAX_POSSIBLE_GAS: int | None = None

  def test_gas_brake_limits_correct(self):
    self.assertIsNotNone(self.MAX_POSSIBLE_BRAKE)
    self.assertIsNotNone(self.MAX_POSSIBLE_GAS)

    self.assertGreater(self.MAX_BRAKE, self.MIN_BRAKE)
    self.assertGreater(self.MAX_GAS, self.MIN_GAS)

  @abc.abstractmethod
  def _send_gas_msg(self, gas: int):
    pass

  @abc.abstractmethod
  def _send_brake_msg(self, brake: int):
    pass

  def test_brake_safety_check(self):
    self._generic_limit_safety_check(self._send_brake_msg, self.MIN_BRAKE, self.MAX_BRAKE, 0, self.MAX_POSSIBLE_BRAKE, 1)

  def test_gas_safety_check(self):
    self._generic_limit_safety_check(self._send_gas_msg, self.MIN_GAS, self.MAX_GAS, self.MIN_POSSIBLE_GAS, self.MAX_POSSIBLE_GAS, 1, self.INACTIVE_GAS)


class TorqueSteeringSafetyTestBase(PandaSafetyTestBase, abc.ABC):

  MAX_RATE_UP = 0
  MAX_RATE_DOWN = 0
  MAX_TORQUE_LOOKUP: tuple[list[float], list[int]] = ([0], [0])
  DYNAMIC_MAX_TORQUE = False
  MAX_RT_DELTA = 0

  NO_STEER_REQ_BIT = False

  @classmethod
  def setUpClass(cls):
    if cls.__name__ == "TorqueSteeringSafetyTestBase":
      cls.safety = None
      raise unittest.SkipTest

  @property
  def MAX_TORQUE(self):
    return max(self.MAX_TORQUE_LOOKUP[1])

  @property
  def _torque_speed_range(self):
    if not self.DYNAMIC_MAX_TORQUE:
      return [0]
    else:
      # test with more precision inside breakpoint range
      min_speed = min(self.MAX_TORQUE_LOOKUP[0])
      max_speed = max(self.MAX_TORQUE_LOOKUP[0])
      return np.concatenate([np.arange(0, min_speed, 5), np.arange(min_speed, max_speed, 0.5), np.arange(max_speed, 40, 5)])

  def _get_max_torque(self, speed):
    # matches safety fudge
    torque = int(np.interp(speed - 1, self.MAX_TORQUE_LOOKUP[0], self.MAX_TORQUE_LOOKUP[1]) + 1)
    return min(torque, self.MAX_TORQUE)

  @abc.abstractmethod
  def _torque_cmd_msg(self, torque, steer_req=1):
    pass

  @abc.abstractmethod
  def _speed_msg(self, speed):
    pass

  def _reset_speed_measurement(self, speed):
    for _ in range(MAX_SAMPLE_VALS):
      self._rx(self._speed_msg(speed))

  def _set_prev_torque(self, t):
    self.safety.set_desired_torque_last(t)
    self.safety.set_rt_torque_last(t)

  def test_steer_safety_check(self):
    for speed in self._torque_speed_range:
      self._reset_speed_measurement(speed)
      max_torque = self._get_max_torque(speed)
      for enabled in [0, 1]:
        for t in range(int(-max_torque * 1.5), int(max_torque * 1.5)):
          self.safety.set_controls_allowed(enabled)
          self._set_prev_torque(t)
          if abs(t) > max_torque or (not enabled and abs(t) > 0):
            self.assertFalse(self._tx(self._torque_cmd_msg(t)))
          else:
            self.assertTrue(self._tx(self._torque_cmd_msg(t)))

  def test_non_realtime_limit_up(self):
    self.safety.set_controls_allowed(True)

    self._set_prev_torque(0)
    self.assertTrue(self._tx(self._torque_cmd_msg(self.MAX_RATE_UP)))
    self._set_prev_torque(0)
    self.assertTrue(self._tx(self._torque_cmd_msg(-self.MAX_RATE_UP)))

    self._set_prev_torque(0)
    self.assertFalse(self._tx(self._torque_cmd_msg(self.MAX_RATE_UP + 1)))
    self.safety.set_controls_allowed(True)
    self._set_prev_torque(0)
    self.assertFalse(self._tx(self._torque_cmd_msg(-self.MAX_RATE_UP - 1)))

  def test_steer_req_bit(self):
    """Asserts all torque safety modes check the steering request bit"""
    if self.NO_STEER_REQ_BIT:
      raise unittest.SkipTest("No steering request bit")

    self.safety.set_controls_allowed(True)
    self._set_prev_torque(self.MAX_TORQUE)

    # Send torque successfully, then only drop the request bit and ensure it stays blocked
    for _ in range(10):
      self.assertTrue(self._tx(self._torque_cmd_msg(self.MAX_TORQUE, 1)))

    self.assertFalse(self._tx(self._torque_cmd_msg(self.MAX_TORQUE, 0)))
    for _ in range(10):
      self.assertFalse(self._tx(self._torque_cmd_msg(self.MAX_TORQUE, 1)))


class SteerRequestCutSafetyTest(TorqueSteeringSafetyTestBase, abc.ABC):

  @classmethod
  def setUpClass(cls):
    if cls.__name__ == "SteerRequestCutSafetyTest":
      cls.safety = None
      raise unittest.SkipTest

  # Safety around steering request bit mismatch tolerance
  MIN_VALID_STEERING_FRAMES: int
  MAX_INVALID_STEERING_FRAMES: int
  STEER_STEP: int = 1

  @property
  def MIN_VALID_STEERING_RT_INTERVAL(self):
    # a ~10% buffer
    return int((self.MIN_VALID_STEERING_FRAMES + 1) * self.STEER_STEP * 10000 * 0.9)

  def test_steer_req_bit_frames(self):
    """
      Certain safety modes implement some tolerance on their steer request bits matching the
      requested torque to avoid a steering fault or lockout and maintain torque. This tests:
        - We can't cut torque for more than one frame
        - We can't cut torque until at least the minimum number of matching steer_req messages
        - We can always recover from violations if steer_req=1
    """

    for min_valid_steer_frames in range(self.MIN_VALID_STEERING_FRAMES * 2):
      # Reset match count and rt timer to allow cut (valid_steer_req_count, ts_steer_req_mismatch_last)
      self.safety.init_tests()
      self.safety.set_timer(self.MIN_VALID_STEERING_RT_INTERVAL)

      # Allow torque cut
      self.safety.set_controls_allowed(True)
      self._set_prev_torque(self.MAX_TORQUE)
      for _ in range(min_valid_steer_frames):
        self.assertTrue(self._tx(self._torque_cmd_msg(self.MAX_TORQUE, steer_req=1)))

      # should tx if we've sent enough valid frames, and we're not cutting torque for too many frames consecutively
      should_tx = min_valid_steer_frames >= self.MIN_VALID_STEERING_FRAMES
      for idx in range(self.MAX_INVALID_STEERING_FRAMES * 2):
        tx = self._tx(self._torque_cmd_msg(self.MAX_TORQUE, steer_req=0))
        self.assertEqual(should_tx and idx < self.MAX_INVALID_STEERING_FRAMES, tx)

      # Keep blocking after one steer_req mismatch
      for _ in range(100):
        self.assertFalse(self._tx(self._torque_cmd_msg(self.MAX_TORQUE, steer_req=0)))

      # Make sure we can recover
      self.assertTrue(self._tx(self._torque_cmd_msg(0, steer_req=1)))
      self._set_prev_torque(self.MAX_TORQUE)
      self.assertTrue(self._tx(self._torque_cmd_msg(self.MAX_TORQUE, steer_req=1)))

  def test_steer_req_bit_multi_invalid(self):
    """
      For safety modes allowing multiple consecutive invalid frames, this ensures that once a valid frame
      is sent after an invalid frame (even without sending the max number of allowed invalid frames),
      all counters are reset.
    """
    for max_invalid_steer_frames in range(1, self.MAX_INVALID_STEERING_FRAMES * 2):
      self.safety.init_tests()
      self.safety.set_timer(self.MIN_VALID_STEERING_RT_INTERVAL)

      # Allow torque cut
      self.safety.set_controls_allowed(True)
      self._set_prev_torque(self.MAX_TORQUE)
      for _ in range(self.MIN_VALID_STEERING_FRAMES):
        self.assertTrue(self._tx(self._torque_cmd_msg(self.MAX_TORQUE, steer_req=1)))

      # Send partial amount of allowed invalid frames
      for idx in range(max_invalid_steer_frames):
        should_tx = idx < self.MAX_INVALID_STEERING_FRAMES
        self.assertEqual(should_tx, self._tx(self._torque_cmd_msg(self.MAX_TORQUE, steer_req=0)))

      # Send one valid frame, and subsequent invalid should now be blocked
      self._set_prev_torque(self.MAX_TORQUE)
      self.assertTrue(self._tx(self._torque_cmd_msg(self.MAX_TORQUE, steer_req=1)))
      for _ in range(self.MIN_VALID_STEERING_FRAMES + 1):
        self.assertFalse(self._tx(self._torque_cmd_msg(self.MAX_TORQUE, steer_req=0)))

  def test_steer_req_bit_realtime(self):
    """
      Realtime safety for cutting steer request bit. This tests:
        - That we allow messages with mismatching steer request bit if time from last is >= MIN_VALID_STEERING_RT_INTERVAL
        - That frame mismatch safety does not interfere with this test
    """
    for rt_us in np.arange(self.MIN_VALID_STEERING_RT_INTERVAL - 50000, self.MIN_VALID_STEERING_RT_INTERVAL + 50000, 10000):
      # Reset match count and rt timer (valid_steer_req_count, ts_steer_req_mismatch_last)
      self.safety.init_tests()

      # Make sure valid_steer_req_count doesn't affect this test
      self.safety.set_controls_allowed(True)
      self._set_prev_torque(self.MAX_TORQUE)
      for _ in range(self.MIN_VALID_STEERING_FRAMES):
        self.assertTrue(self._tx(self._torque_cmd_msg(self.MAX_TORQUE, steer_req=1)))

      # Normally, sending MIN_VALID_STEERING_FRAMES valid frames should always allow
      self.safety.set_timer(max(rt_us, 0))
      should_tx = rt_us >= self.MIN_VALID_STEERING_RT_INTERVAL
      for _ in range(self.MAX_INVALID_STEERING_FRAMES):
        self.assertEqual(should_tx, self._tx(self._torque_cmd_msg(self.MAX_TORQUE, steer_req=0)))

      # Keep blocking after one steer_req mismatch
      for _ in range(100):
        self.assertFalse(self._tx(self._torque_cmd_msg(self.MAX_TORQUE, steer_req=0)))

      # Make sure we can recover
      self.assertTrue(self._tx(self._torque_cmd_msg(0, steer_req=1)))
      self._set_prev_torque(self.MAX_TORQUE)
      self.assertTrue(self._tx(self._torque_cmd_msg(self.MAX_TORQUE, steer_req=1)))


class DriverTorqueSteeringSafetyTest(TorqueSteeringSafetyTestBase, abc.ABC):

  DRIVER_TORQUE_ALLOWANCE = 0
  DRIVER_TORQUE_FACTOR = 0

  @classmethod
  def setUpClass(cls):
    if cls.__name__ == "DriverTorqueSteeringSafetyTest":
      cls.safety = None
      raise unittest.SkipTest

  @abc.abstractmethod
  def _torque_driver_msg(self, torque):
    pass

  def _reset_torque_driver_measurement(self, torque):
    for _ in range(MAX_SAMPLE_VALS):
      self._rx(self._torque_driver_msg(torque))

  def test_non_realtime_limit_up(self):
    self._reset_torque_driver_measurement(0)
    super().test_non_realtime_limit_up()

  def test_against_torque_driver(self):
    # Tests down limits and driver torque blending
    self.safety.set_controls_allowed(True)

    for speed in self._torque_speed_range:
      self._reset_speed_measurement(speed)
      max_torque = self._get_max_torque(speed)

      # Cannot stay at MAX_TORQUE if above DRIVER_TORQUE_ALLOWANCE
      for sign in [-1, 1]:
        for driver_torque in np.arange(0, self.DRIVER_TORQUE_ALLOWANCE * 2, 1):
          self._reset_torque_driver_measurement(-driver_torque * sign)
          self._set_prev_torque(max_torque * sign)
          should_tx = abs(driver_torque) <= self.DRIVER_TORQUE_ALLOWANCE
          self.assertEqual(should_tx, self._tx(self._torque_cmd_msg(max_torque * sign)))

      # arbitrary high driver torque to ensure max steer torque is allowed
      max_driver_torque = int(max_torque / self.DRIVER_TORQUE_FACTOR + self.DRIVER_TORQUE_ALLOWANCE + 1)

      # spot check some individual cases
      for sign in [-1, 1]:
        # Ensure we wind down factor units for every unit above allowance
        driver_torque = (self.DRIVER_TORQUE_ALLOWANCE + 10) * sign
        torque_desired = (max_torque - 10 * self.DRIVER_TORQUE_FACTOR) * sign
        delta = 1 * sign
        self._set_prev_torque(torque_desired)
        self._reset_torque_driver_measurement(-driver_torque)
        self.assertTrue(self._tx(self._torque_cmd_msg(torque_desired)))
        self._set_prev_torque(torque_desired + delta)
        self._reset_torque_driver_measurement(-driver_torque)
        self.assertFalse(self._tx(self._torque_cmd_msg(torque_desired + delta)))

        # If we're well past the allowance, minimum wind down is MAX_RATE_DOWN
        self._set_prev_torque(max_torque * sign)
        self._reset_torque_driver_measurement(-max_driver_torque * sign)
        self.assertTrue(self._tx(self._torque_cmd_msg((max_torque - self.MAX_RATE_DOWN) * sign)))
        self._set_prev_torque(max_torque * sign)
        self._reset_torque_driver_measurement(-max_driver_torque * sign)
        self.assertTrue(self._tx(self._torque_cmd_msg(0)))
        self._set_prev_torque(max_torque * sign)
        self._reset_torque_driver_measurement(-max_driver_torque * sign)
        self.assertFalse(self._tx(self._torque_cmd_msg((max_torque - self.MAX_RATE_DOWN + 1) * sign)))

  def test_realtime_limits(self):
    self.safety.set_controls_allowed(True)

    for sign in [-1, 1]:
      self.safety.init_tests()
      self._set_prev_torque(0)
      self._reset_torque_driver_measurement(0)
      for t in np.arange(0, self.MAX_RT_DELTA, 1):
        t *= sign
        self.assertTrue(self._tx(self._torque_cmd_msg(t)))
      self.assertFalse(self._tx(self._torque_cmd_msg(sign * (self.MAX_RT_DELTA + 1))))

      self._set_prev_torque(0)
      for t in np.arange(0, self.MAX_RT_DELTA, 1):
        t *= sign
        self.assertTrue(self._tx(self._torque_cmd_msg(t)))

      # Increase timer to update rt_torque_last
      self.safety.set_timer(RT_INTERVAL + 1)
      self.assertTrue(self._tx(self._torque_cmd_msg(sign * (self.MAX_RT_DELTA - 1))))
      self.assertTrue(self._tx(self._torque_cmd_msg(sign * (self.MAX_RT_DELTA + 1))))

  def test_reset_driver_torque_measurements(self):
    # Tests that the driver torque measurement sample_t is reset on safety mode init
    for t in np.linspace(-self.MAX_TORQUE, self.MAX_TORQUE, MAX_SAMPLE_VALS):
      self.assertTrue(self._rx(self._torque_driver_msg(t)))

    self.assertNotEqual(self.safety.get_torque_driver_min(), 0)
    self.assertNotEqual(self.safety.get_torque_driver_max(), 0)

    self._reset_safety_hooks()
    self.assertEqual(self.safety.get_torque_driver_min(), 0)
    self.assertEqual(self.safety.get_torque_driver_max(), 0)


class MotorTorqueSteeringSafetyTest(TorqueSteeringSafetyTestBase, abc.ABC):

  MAX_TORQUE_ERROR = 0
  TORQUE_MEAS_TOLERANCE = 0

  @classmethod
  def setUpClass(cls):
    if cls.__name__ == "MotorTorqueSteeringSafetyTest":
      cls.safety = None
      raise unittest.SkipTest

  @abc.abstractmethod
  def _torque_meas_msg(self, torque):
    pass

  def _set_prev_torque(self, t):
    super()._set_prev_torque(t)
    self.safety.set_torque_meas(t, t)

  def test_torque_absolute_limits(self):
    for speed in self._torque_speed_range:
      self._reset_speed_measurement(speed)
      max_torque = self._get_max_torque(speed)
      for controls_allowed in [True, False]:
        for torque in np.arange(-max_torque - 1000, max_torque + 1000, self.MAX_RATE_UP):
          self.safety.set_controls_allowed(controls_allowed)
          self.safety.set_rt_torque_last(torque)
          self.safety.set_torque_meas(torque, torque)
          self.safety.set_desired_torque_last(torque - self.MAX_RATE_UP)

          if controls_allowed:
            send = (-max_torque <= torque <= max_torque)
          else:
            send = torque == 0

          self.assertEqual(send, self._tx(self._torque_cmd_msg(torque)))

  def test_non_realtime_limit_down(self):
    self.safety.set_controls_allowed(True)

    for speed in self._torque_speed_range:
      self._reset_speed_measurement(speed)
      max_torque = self._get_max_torque(speed)

      torque_meas = max_torque - self.MAX_TORQUE_ERROR - 50

      self.safety.set_rt_torque_last(max_torque)
      self.safety.set_torque_meas(torque_meas, torque_meas)
      self.safety.set_desired_torque_last(max_torque)
      self.assertTrue(self._tx(self._torque_cmd_msg(max_torque - self.MAX_RATE_DOWN)))

      self.safety.set_rt_torque_last(max_torque)
      self.safety.set_torque_meas(torque_meas, torque_meas)
      self.safety.set_desired_torque_last(max_torque)
      self.assertFalse(self._tx(self._torque_cmd_msg(max_torque - self.MAX_RATE_DOWN + 1)))

  def test_exceed_torque_sensor(self):
    self.safety.set_controls_allowed(True)

    for sign in [-1, 1]:
      self._set_prev_torque(0)
      for t in np.arange(0, self.MAX_TORQUE_ERROR + 2, 2):  # step needs to be smaller than MAX_TORQUE_ERROR
        t *= sign
        self.assertTrue(self._tx(self._torque_cmd_msg(t)))

      self.assertFalse(self._tx(self._torque_cmd_msg(sign * (self.MAX_TORQUE_ERROR + 2))))

  def test_realtime_limit_up(self):
    self.safety.set_controls_allowed(True)

    for sign in [-1, 1]:
      self.safety.init_tests()
      self._set_prev_torque(0)
      for t in np.arange(0, self.MAX_RT_DELTA + 1, 1):
        t *= sign
        self.safety.set_torque_meas(t, t)
        self.assertTrue(self._tx(self._torque_cmd_msg(t)))
      self.assertFalse(self._tx(self._torque_cmd_msg(sign * (self.MAX_RT_DELTA + 1))))

      self._set_prev_torque(0)
      for t in np.arange(0, self.MAX_RT_DELTA + 1, 1):
        t *= sign
        self.safety.set_torque_meas(t, t)
        self.assertTrue(self._tx(self._torque_cmd_msg(t)))

      # Increase timer to update rt_torque_last
      self.safety.set_timer(RT_INTERVAL + 1)
      self.assertTrue(self._tx(self._torque_cmd_msg(sign * self.MAX_RT_DELTA)))
      self.assertTrue(self._tx(self._torque_cmd_msg(sign * (self.MAX_RT_DELTA + 1))))

  def test_torque_measurements(self):
    trq = 50
    for t in [trq, -trq, 0, 0, 0, 0]:
      self._rx(self._torque_meas_msg(t))

    max_range = range(trq, trq + self.TORQUE_MEAS_TOLERANCE + 1)
    min_range = range(-(trq + self.TORQUE_MEAS_TOLERANCE), -trq + 1)
    self.assertTrue(self.safety.get_torque_meas_min() in min_range)
    self.assertTrue(self.safety.get_torque_meas_max() in max_range)

    max_range = range(self.TORQUE_MEAS_TOLERANCE + 1)
    min_range = range(-(trq + self.TORQUE_MEAS_TOLERANCE), -trq + 1)
    self._rx(self._torque_meas_msg(0))
    self.assertTrue(self.safety.get_torque_meas_min() in min_range)
    self.assertTrue(self.safety.get_torque_meas_max() in max_range)

    max_range = range(self.TORQUE_MEAS_TOLERANCE + 1)
    min_range = range(-self.TORQUE_MEAS_TOLERANCE, 0 + 1)
    self._rx(self._torque_meas_msg(0))
    self.assertTrue(self.safety.get_torque_meas_min() in min_range)
    self.assertTrue(self.safety.get_torque_meas_max() in max_range)

  def test_reset_torque_measurements(self):
    # Tests that the torque measurement sample_t is reset on safety mode init
    for t in np.linspace(-self.MAX_TORQUE, self.MAX_TORQUE, MAX_SAMPLE_VALS):
      self.assertTrue(self._rx(self._torque_meas_msg(t)))

    self.assertNotEqual(self.safety.get_torque_meas_min(), 0)
    self.assertNotEqual(self.safety.get_torque_meas_max(), 0)

    self._reset_safety_hooks()
    self.assertEqual(self.safety.get_torque_meas_min(), 0)
    self.assertEqual(self.safety.get_torque_meas_max(), 0)


class VehicleSpeedSafetyTest(PandaSafetyTestBase):
  @classmethod
  def setUpClass(cls):
    if cls.__name__ == "VehicleSpeedSafetyTest":
      cls.safety = None
      raise unittest.SkipTest

  @abc.abstractmethod
  def _speed_msg(self, speed):
    pass

  def test_vehicle_speed_measurements(self):
    # TODO: lower tolerance on these tests
    self._common_measurement_test(self._speed_msg, 0, 80, 1, self.safety.get_vehicle_speed_min, self.safety.get_vehicle_speed_max)


class AngleSteeringSafetyTest(VehicleSpeedSafetyTest):

  STEER_ANGLE_MAX: float = 300
  STEER_ANGLE_TEST_MAX: float = None
  DEG_TO_CAN: float
  ANGLE_RATE_BP: list[float]
  ANGLE_RATE_UP: list[float]  # windup limit
  ANGLE_RATE_DOWN: list[float]  # unwind limit

  # Real time limits
  LATERAL_FREQUENCY: int = -1  # Hz

  @classmethod
  def setUpClass(cls):
    if cls.__name__ == "AngleSteeringSafetyTest":
      cls.safety = None
      raise unittest.SkipTest

  @abc.abstractmethod
  def _angle_cmd_msg(self, angle: float, enabled: bool, increment_timer: bool = True):
    pass

  @abc.abstractmethod
  def _angle_meas_msg(self, angle: float):
    pass

  def _get_steer_cmd_angle_max(self, speed):
    return self.STEER_ANGLE_MAX

  def _set_prev_desired_angle(self, t):
    t = round(t * self.DEG_TO_CAN)
    self.safety.set_desired_angle_last(t)

  def _reset_angle_measurement(self, angle):
    for _ in range(MAX_SAMPLE_VALS):
      self._rx(self._angle_meas_msg(angle))

  def _reset_speed_measurement(self, speed):
    for _ in range(MAX_SAMPLE_VALS):
      self._rx(self._speed_msg(speed))

  def test_steering_angle_measurements(self):
    self._common_measurement_test(self._angle_meas_msg, -self.STEER_ANGLE_MAX, self.STEER_ANGLE_MAX, self.DEG_TO_CAN,
                                  self.safety.get_angle_meas_min, self.safety.get_angle_meas_max)

  def test_angle_cmd_when_enabled(self):
    # when controls are allowed, angle cmd rate limit is enforced
    speeds = [0., 1., 5., 10., 15., 50.]
    # TODO: what should CANPacker do here? we should also have good coverage checks on this
    if self.STEER_ANGLE_TEST_MAX is None:
        self.STEER_ANGLE_TEST_MAX = self.STEER_ANGLE_MAX * 2
    angles = np.concatenate((np.arange(-self.STEER_ANGLE_TEST_MAX, self.STEER_ANGLE_TEST_MAX, 5), [0]))
    for a in angles:
      for s in speeds:
        max_delta_up = np.interp(s, self.ANGLE_RATE_BP, self.ANGLE_RATE_UP)
        max_delta_down = np.interp(s, self.ANGLE_RATE_BP, self.ANGLE_RATE_DOWN)

        # first test against false positives
        self._reset_angle_measurement(a)
        self._reset_speed_measurement(s)

        self._set_prev_desired_angle(a)
        self.safety.set_controls_allowed(1)

        # Stay within limits
        # Up
        self.assertTrue(self._tx(self._angle_cmd_msg(a + sign_of(a) * max_delta_up, True)))
        self.assertTrue(self.safety.get_controls_allowed())

        # Don't change
        self.assertTrue(self._tx(self._angle_cmd_msg(a, True)))
        self.assertTrue(self.safety.get_controls_allowed())

        # Down
        self.assertTrue(self._tx(self._angle_cmd_msg(a - sign_of(a) * max_delta_down, True)))
        self.assertTrue(self.safety.get_controls_allowed())

        # Inject too high rates
        # Up
        self.assertFalse(self._tx(self._angle_cmd_msg(a + sign_of(a) * (max_delta_up + 1.1), True)))

        # Don't change
        self.safety.set_controls_allowed(1)
        self._set_prev_desired_angle(a)
        self.assertTrue(self.safety.get_controls_allowed())
        self.assertTrue(self._tx(self._angle_cmd_msg(a, True)))
        self.assertTrue(self.safety.get_controls_allowed())

        # Down
        self.assertFalse(self._tx(self._angle_cmd_msg(a - sign_of(a) * (max_delta_down + 1.1), True)))

        # Check desired steer should be the same as steer angle when controls are off
        self.safety.set_controls_allowed(0)
        should_tx = abs(a) <= abs(self.STEER_ANGLE_MAX)
        self.assertEqual(should_tx, self._tx(self._angle_cmd_msg(a, False)))

  def test_angle_cmd_when_disabled(self):
    # Tests that only angles close to the meas are allowed while
    # steer actuation bit is 0, regardless of controls allowed.
    for controls_allowed in (True, False):
      self.safety.set_controls_allowed(controls_allowed)

      for steer_control_enabled in (True, False):
        for angle_meas in np.arange(-90, 91, 10):
          self._reset_angle_measurement(angle_meas)

          for angle_cmd in np.arange(-90, 91, 10):
            self._set_prev_desired_angle(angle_cmd)

            # controls_allowed is checked if actuation bit is 1, else the angle must be close to meas (inactive)
            should_tx = controls_allowed if steer_control_enabled else angle_cmd == angle_meas
            self.assertEqual(should_tx, self._tx(self._angle_cmd_msg(angle_cmd, steer_control_enabled)))

  def test_angle_violation(self):
    # If violation occurs, angle cmd is blocked until reset to 0. Matches behavior of torque safety modes
    self.safety.set_controls_allowed(True)

    for speed in (0., 1., 5., 10., 15., 50.):
      self._tx(self._angle_cmd_msg(0, True))
      self._reset_speed_measurement(speed)

      for _ in range(20):
        self.assertFalse(self._tx(self._angle_cmd_msg(self._get_steer_cmd_angle_max(speed), True)))
      self.assertTrue(self._tx(self._angle_cmd_msg(0, True)))

  def test_rt_limits(self):
    # TODO: remove and check all safety modes
    if self.LATERAL_FREQUENCY == -1:
      raise unittest.SkipTest("No real time limits")

    # Angle safety enforces real time limits by checking the message send frequency in a 250ms time window
    self.safety.set_timer(0)
    self.safety.set_controls_allowed(True)
    max_rt_msgs = int(self.LATERAL_FREQUENCY * RT_INTERVAL / 1e6 * 1.2 + 1)  # 1.2x buffer

    for i in range(max_rt_msgs * 2):
      should_tx = i <= max_rt_msgs
      self.assertEqual(should_tx, self._tx(self._angle_cmd_msg(0, True, increment_timer=False)))

    # One under RT interval should do nothing
    self.safety.set_timer(RT_INTERVAL - 1)
    for _ in range(5):
      self.assertFalse(self._tx(self._angle_cmd_msg(0, True, increment_timer=False)))

    # Increment timer and send 1 message to reset RT window
    self.safety.set_timer(RT_INTERVAL)
    self.assertFalse(self._tx(self._angle_cmd_msg(0, True, increment_timer=False)))
    for _ in range(5):
      self.assertTrue(self._tx(self._angle_cmd_msg(0, True, increment_timer=False)))


class PandaSafetyTest(PandaSafetyTestBase):
  TX_MSGS: list[list[int]] | None = None
  SCANNED_ADDRS = [*range(0x800),                      # Entire 11-bit CAN address space
                   *range(0x18DA00F1, 0x18DB00F1, 0x100),   # 29-bit UDS physical addressing
                   *range(0x18DB00F1, 0x18DC00F1, 0x100),   # 29-bit UDS functional addressing
                   *range(0x3300, 0x3400)]                  # Honda
  FWD_BLACKLISTED_ADDRS: dict[int, list[int]] = {}  # {bus: [addr]}
  FWD_BUS_LOOKUP: dict[int, int] = {0: 2, 2: 0}

  @classmethod
  def setUpClass(cls):
    if cls.__name__ == "PandaSafetyTest" or cls.__name__.endswith('Base'):
      cls.safety = None
      raise unittest.SkipTest

  # ***** standard tests for all safety modes *****

  def test_tx_msg_in_scanned_range(self):
    # the relay malfunction, fwd hook, and spam can tests don't exhaustively
    # scan the entire 29-bit address space, only some known important ranges
    # make sure SCANNED_ADDRS stays up to date with car port TX_MSGS; new
    # model ports should expand the range if needed
    for msg in self.TX_MSGS:
      self.assertTrue(msg[0] in self.SCANNED_ADDRS, f"{msg[0]=:#x}")

  def test_fwd_hook(self):
    # some safety modes don't forward anything, while others blacklist msgs
    for bus in range(3):
      for addr in self.SCANNED_ADDRS:
        # assume len 8
        fwd_bus = self.FWD_BUS_LOOKUP.get(bus, -1)
        if bus in self.FWD_BLACKLISTED_ADDRS and addr in self.FWD_BLACKLISTED_ADDRS[bus]:
          fwd_bus = -1
        self.assertEqual(fwd_bus, self.safety.safety_fwd_hook(bus, addr), f"{addr=:#x} from {bus=} to {fwd_bus=}")

  def test_spam_can_buses(self):
    for bus in range(4):
      for addr in self.SCANNED_ADDRS:
        if [addr, bus] not in self.TX_MSGS:
          self.assertFalse(self._tx(make_msg(bus, addr, 8)), f"allowed TX {addr=} {bus=}")

  def test_default_controls_not_allowed(self):
    self.assertFalse(self.safety.get_controls_allowed())

  def test_manually_enable_controls_allowed(self):
    self.safety.set_controls_allowed(1)
    self.assertTrue(self.safety.get_controls_allowed())
    self.safety.set_controls_allowed(0)
    self.assertFalse(self.safety.get_controls_allowed())

  def test_tx_hook_on_wrong_safety_mode(self):
    files = os.listdir(os.path.dirname(os.path.realpath(__file__)))
    test_files = [f for f in files if f.startswith("test_") and f.endswith(".py")]

    current_test = self.__class__.__name__

    all_tx = []
    for tf in test_files:
      test = importlib.import_module("opendbc.safety.tests."+tf[:-3])
      for attr in dir(test):
        if attr.startswith("Test") and attr != current_test:
          tc = getattr(test, attr)
          tx = tc.TX_MSGS
          if tx is not None and not attr.endswith('Base'):
            # No point in comparing different Tesla safety modes
            if 'Tesla' in attr and 'Tesla' in current_test:
              continue
            # No point in comparing to ALLOUTPUT which allows all messages
            if attr.startswith('TestAllOutput'):
              continue
            if attr.startswith('TestToyota') and current_test.startswith('TestToyota'):
              continue
            if attr.startswith('TestSubaruGen') and current_test.startswith('TestSubaruGen'):
              continue
            if attr.startswith('TestSubaruPreglobal') and current_test.startswith('TestSubaruPreglobal'):
              continue
            if {attr, current_test}.issubset({'TestVolkswagenPqSafety', 'TestVolkswagenPqStockSafety', 'TestVolkswagenPqLongSafety'}):
              continue
            if {attr, current_test}.issubset({'TestGmCameraSafety', 'TestGmCameraLongitudinalSafety', 'TestGmAscmSafety',
                                              'TestGmCameraEVSafety', 'TestGmCameraLongitudinalEVSafety', 'TestGmAscmEVSafety'}):
              continue
            if attr.startswith('TestFord') and current_test.startswith('TestFord'):
              continue
            if attr.startswith('TestHyundaiCanfd') and current_test.startswith('TestHyundaiCanfd'):
              continue
            if {attr, current_test}.issubset({'TestHyundaiLongitudinalSafety', 'TestHyundaiLongitudinalSafetyCameraSCC', 'TestHyundaiSafetyFCEVLong'}):
              continue
            if {attr, current_test}.issubset({'TestVolkswagenMqbSafety', 'TestVolkswagenMqbStockSafety', 'TestVolkswagenMqbLongSafety'}):
              continue

            # overlapping TX addrs, but they're not actuating messages for either car
            if attr == 'TestHyundaiCanfdLKASteeringLongEV' and current_test.startswith('TestToyota'):
              tx = list(filter(lambda m: m[0] not in [0x160, ], tx))

            # Volkswagen MQB longitudinal actuating message overlaps with the Subaru lateral actuating message
            if attr == 'TestVolkswagenMqbLongSafety' and current_test.startswith('TestSubaru'):
              tx = list(filter(lambda m: m[0] not in [0x122, ], tx))

            # Volkswagen MQB and Honda Nidec ACC HUD messages overlap
            if attr == 'TestVolkswagenMqbLongSafety' and current_test.startswith('TestHondaNidec'):
              tx = list(filter(lambda m: m[0] not in [0x30c, ], tx))

            # Volkswagen MQB and Honda Bosch Radarless ACC HUD messages overlap
            if attr == 'TestVolkswagenMqbLongSafety' and current_test.startswith('TestHondaBoschRadarless'):
              tx = list(filter(lambda m: m[0] not in [0x30c, ], tx))

            # TODO: Temporary, should be fixed in panda firmware, safety_honda.h
            if attr.startswith('TestHonda'):
              # exceptions for common msgs across different hondas
              tx = list(filter(lambda m: m[0] not in [0x1FA, 0x30C, 0x33D, 0x33DB], tx))

            if attr.startswith('TestHyundaiLongitudinal'):
              # exceptions for common msgs across different Hyundai CAN platforms
              tx = list(filter(lambda m: m[0] not in [0x420, 0x50A, 0x389, 0x4A2], tx))
            all_tx.append([[m[0], m[1], attr] for m in tx])

    # make sure we got all the msgs
    self.assertTrue(len(all_tx) >= len(test_files)-1)

    for tx_msgs in all_tx:
      for addr, bus, test_name in tx_msgs:
        msg = make_msg(bus, addr)
        self.safety.set_controls_allowed(1)
        # TODO: this should be blocked
        if current_test in ["TestNissanSafety", "TestNissanSafetyAltEpsBus", "TestNissanLeafSafety"] and [addr, bus] in self.TX_MSGS:
          continue
        self.assertFalse(self._tx(msg), f"transmit of {addr=:#x} {bus=} from {test_name} during {current_test} was allowed")


@add_regen_tests
class PandaCarSafetyTest(PandaSafetyTest):
  STANDSTILL_THRESHOLD: float = 0.0
  GAS_PRESSED_THRESHOLD = 0
  RELAY_MALFUNCTION_ADDRS: dict[int, tuple[int, ...]] | None = None

  @classmethod
  def setUpClass(cls):
    if cls.__name__ == "PandaCarSafetyTest" or cls.__name__.endswith('Base'):
      cls.safety = None
      raise unittest.SkipTest

  @abc.abstractmethod
  def _user_brake_msg(self, brake):
    pass

  def _user_regen_msg(self, regen):
    pass

  @abc.abstractmethod
  def _speed_msg(self, speed):
    pass

  @abc.abstractmethod
  def _speed_msg_2(self, speed: float):
    pass

  # Safety modes can override if vehicle_moving is driven by a different message
  def _vehicle_moving_msg(self, speed: float):
    return self._speed_msg(speed)

  @abc.abstractmethod
  def _user_gas_msg(self, gas):
    pass

  @abc.abstractmethod
  def _pcm_status_msg(self, enable):
    pass

  # ***** standard tests for all car-specific safety modes *****

  def test_relay_malfunction(self):
    # each car has an addr that is used to detect relay malfunction
    # if that addr is seen on specified bus, triggers the relay malfunction
    # protection logic: both tx_hook and fwd_hook are expected to return failure
    self.assertFalse(self.safety.get_relay_malfunction())
    for bus in range(3):
      for addr in self.SCANNED_ADDRS:
        self.safety.set_relay_malfunction(False)
        self._rx(make_msg(bus, addr, 8))
        should_relay_malfunction = addr in self.RELAY_MALFUNCTION_ADDRS.get(bus, ())
        self.assertEqual(should_relay_malfunction, self.safety.get_relay_malfunction(), (bus, hex(addr)))

    # test relay malfunction protection logic
    self.safety.set_relay_malfunction(True)
    for bus in range(3):
      for addr in self.SCANNED_ADDRS:
        self.assertFalse(self._tx(make_msg(bus, addr, 8)))
        self.assertEqual(-1, self.safety.safety_fwd_hook(bus, addr))

  def test_prev_gas(self):
    self.assertFalse(self.safety.get_gas_pressed_prev())
    for pressed in [self.GAS_PRESSED_THRESHOLD + 1, 0]:
      self._rx(self._user_gas_msg(pressed))
      self.assertEqual(bool(pressed), self.safety.get_gas_pressed_prev())

  def test_allow_engage_with_gas_pressed(self):
    self._rx(self._user_gas_msg(1))
    self.safety.set_controls_allowed(True)
    self._rx(self._user_gas_msg(1))
    self.assertTrue(self.safety.get_controls_allowed())
    self._rx(self._user_gas_msg(1))
    self.assertTrue(self.safety.get_controls_allowed())

  def test_no_disengage_on_gas(self):
    self._rx(self._user_gas_msg(0))
    self.safety.set_controls_allowed(True)
    self._rx(self._user_gas_msg(self.GAS_PRESSED_THRESHOLD + 1))
    # Test we allow lateral, but not longitudinal
    self.assertTrue(self.safety.get_controls_allowed())
    self.assertFalse(self.safety.get_longitudinal_allowed())
    # Make sure we can re-gain longitudinal actuation
    self._rx(self._user_gas_msg(0))
    self.assertTrue(self.safety.get_longitudinal_allowed())

  def test_prev_user_brake(self, _user_brake_msg=None, get_brake_pressed_prev=None):
    if _user_brake_msg is None:
      _user_brake_msg = self._user_brake_msg
      get_brake_pressed_prev = self.safety.get_brake_pressed_prev

    self.assertFalse(get_brake_pressed_prev())
    for pressed in [True, False]:
      self._rx(_user_brake_msg(not pressed))
      self.assertEqual(not pressed, get_brake_pressed_prev())
      self._rx(_user_brake_msg(pressed))
      self.assertEqual(pressed, get_brake_pressed_prev())

  def test_enable_control_allowed_from_cruise(self):
    self._rx(self._pcm_status_msg(False))
    self.assertFalse(self.safety.get_controls_allowed())
    self._rx(self._pcm_status_msg(True))
    self.assertTrue(self.safety.get_controls_allowed())

  def test_disable_control_allowed_from_cruise(self):
    self.safety.set_controls_allowed(1)
    self._rx(self._pcm_status_msg(False))
    self.assertFalse(self.safety.get_controls_allowed())

  def test_cruise_engaged_prev(self):
    for engaged in [True, False]:
      self._rx(self._pcm_status_msg(engaged))
      self.assertEqual(engaged, self.safety.get_cruise_engaged_prev())
      self._rx(self._pcm_status_msg(not engaged))
      self.assertEqual(not engaged, self.safety.get_cruise_engaged_prev())

  def test_allow_user_brake_at_zero_speed(self, _user_brake_msg=None, get_brake_pressed_prev=None):
    if _user_brake_msg is None:
      _user_brake_msg = self._user_brake_msg

    # Brake was already pressed
    self._rx(self._vehicle_moving_msg(0))
    self._rx(_user_brake_msg(1))
    self.safety.set_controls_allowed(1)
    self._rx(_user_brake_msg(1))
    self.assertTrue(self.safety.get_controls_allowed())
    self.assertTrue(self.safety.get_longitudinal_allowed())
    self._rx(_user_brake_msg(0))
    self.assertTrue(self.safety.get_controls_allowed())
    self.assertTrue(self.safety.get_longitudinal_allowed())
    # rising edge of brake should disengage
    self._rx(_user_brake_msg(1))
    self.assertFalse(self.safety.get_controls_allowed())
    self.assertFalse(self.safety.get_longitudinal_allowed())
    self._rx(_user_brake_msg(0))  # reset no brakes

  def test_not_allow_user_brake_when_moving(self, _user_brake_msg=None, get_brake_pressed_prev=None):
    if _user_brake_msg is None:
      _user_brake_msg = self._user_brake_msg

    # Brake was already pressed
    self._rx(_user_brake_msg(1))
    self.safety.set_controls_allowed(1)
    self._rx(self._vehicle_moving_msg(self.STANDSTILL_THRESHOLD))
    self._rx(_user_brake_msg(1))
    self.assertTrue(self.safety.get_controls_allowed())
    self.assertTrue(self.safety.get_longitudinal_allowed())
    self._rx(self._vehicle_moving_msg(self.STANDSTILL_THRESHOLD + 1))
    self._rx(_user_brake_msg(1))
    self.assertFalse(self.safety.get_controls_allowed())
    self.assertFalse(self.safety.get_longitudinal_allowed())
    self._rx(self._vehicle_moving_msg(0))

  def test_vehicle_moving(self):
    self.assertFalse(self.safety.get_vehicle_moving())

    # not moving
    self._rx(self._vehicle_moving_msg(0))
    self.assertFalse(self.safety.get_vehicle_moving())

    # speed is at threshold
    self._rx(self._vehicle_moving_msg(self.STANDSTILL_THRESHOLD))
    self.assertFalse(self.safety.get_vehicle_moving())

    # past threshold
    self._rx(self._vehicle_moving_msg(self.STANDSTILL_THRESHOLD + 1))
    self.assertTrue(self.safety.get_vehicle_moving())

  def test_rx_hook_speed_mismatch(self):
    if self._speed_msg_2(0) is None:
      raise unittest.SkipTest("No second speed message for this safety mode")

    for speed in np.arange(0, 40, 0.5):
      for speed_delta in np.arange(-5, 5, 0.1):
        speed_2 = round(max(speed + speed_delta, 0), 1)
        # Set controls allowed in between rx since first message can reset it
        self._rx(self._speed_msg(speed))
        self.safety.set_controls_allowed(True)
        self._rx(self._speed_msg_2(speed_2))

        within_delta = abs(speed - speed_2) <= MAX_SPEED_DELTA
        self.assertEqual(self.safety.get_controls_allowed(), within_delta)

  def test_safety_tick(self):
    self.safety.set_timer(int(2e6))
    self.safety.set_controls_allowed(True)
    self.safety.safety_tick_current_safety_config()
    self.assertFalse(self.safety.get_controls_allowed())
    self.assertFalse(self.safety.safety_config_valid())
