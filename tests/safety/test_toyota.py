#!/usr/bin/env python3
import unittest
import numpy as np
import libpandasafety_py  # pylint: disable=import-error
from panda import Panda

MAX_RATE_UP = 10
MAX_RATE_DOWN = 25
MAX_TORQUE = 1500

MAX_ACCEL = 1500
MIN_ACCEL = -3000

MAX_RT_DELTA = 375
RT_INTERVAL = 250000

MAX_TORQUE_ERROR = 350

INTERCEPTOR_THRESHOLD = 475

def twos_comp(val, bits):
  if val >= 0:
    return val
  else:
    return (2**bits) + val

def sign(a):
  if a > 0:
    return 1
  else:
    return -1

class TestToyotaSafety(unittest.TestCase):
  @classmethod
  def setUp(cls):
    cls.safety = libpandasafety_py.libpandasafety
    cls.safety.safety_set_mode(Panda.SAFETY_TOYOTA, 100)
    cls.safety.init_tests_toyota()

  def _send_msg(self, bus, addr, length):
    to_send = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
    to_send[0].RIR = addr << 21
    to_send[0].RDTR = length
    to_send[0].RDTR = bus << 4
    return to_send

  def _set_prev_torque(self, t):
    self.safety.set_toyota_desired_torque_last(t)
    self.safety.set_toyota_rt_torque_last(t)
    self.safety.set_toyota_torque_meas(t, t)

  def _torque_meas_msg(self, torque):
    to_send = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
    to_send[0].RIR = 0x260 << 21

    t = twos_comp(torque, 16)
    to_send[0].RDHR = t | ((t & 0xFF) << 16)
    return to_send

  def _torque_msg(self, torque):
    to_send = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
    to_send[0].RIR = 0x2E4 << 21

    t = twos_comp(torque, 16)
    to_send[0].RDLR = t | ((t & 0xFF) << 16)
    return to_send

  def _accel_msg(self, accel):
    to_send = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
    to_send[0].RIR = 0x343 << 21

    a = twos_comp(accel, 16)
    to_send[0].RDLR = (a & 0xFF) << 8 | (a >> 8)
    return to_send

  def _send_gas_msg(self, gas):
    to_send = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
    to_send[0].RIR = 0x2C1 << 21
    to_send[0].RDHR = (gas & 0xFF) << 16

    return to_send

  def _send_interceptor_msg(self, gas, addr):
    to_send = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
    to_send[0].RIR = addr << 21
    to_send[0].RDTR = 6
    gas2 = gas * 2
    to_send[0].RDLR = ((gas & 0xff) << 8) | ((gas & 0xff00) >> 8) | \
                      ((gas2 & 0xff) << 24) | ((gas2 & 0xff00) << 8)

    return to_send

  def _pcm_cruise_msg(self, cruise_on):
    to_send = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
    to_send[0].RIR = 0x1D2 << 21
    to_send[0].RDLR = cruise_on << 5

    return to_send

  def test_default_controls_not_allowed(self):
    self.assertFalse(self.safety.get_controls_allowed())

  def test_manually_enable_controls_allowed(self):
    self.safety.set_controls_allowed(1)
    self.assertTrue(self.safety.get_controls_allowed())

  def test_enable_control_allowed_from_cruise(self):
    self.safety.safety_rx_hook(self._pcm_cruise_msg(False))
    self.assertFalse(self.safety.get_controls_allowed())
    self.safety.safety_rx_hook(self._pcm_cruise_msg(True))
    self.assertTrue(self.safety.get_controls_allowed())

  def test_disable_control_allowed_from_cruise(self):
    self.safety.set_controls_allowed(1)
    self.safety.safety_rx_hook(self._pcm_cruise_msg(False))
    self.assertFalse(self.safety.get_controls_allowed())

  def test_prev_gas(self):
    for g in range(0, 256):
      self.safety.safety_rx_hook(self._send_gas_msg(g))
      self.assertEqual(g, self.safety.get_toyota_gas_prev())

  def test_prev_gas_interceptor(self):
    self.safety.safety_rx_hook(self._send_interceptor_msg(0x0, 0x201))
    self.assertFalse(self.safety.get_gas_interceptor_prev())
    self.safety.safety_rx_hook(self._send_interceptor_msg(0x1000, 0x201))
    self.assertTrue(self.safety.get_gas_interceptor_prev())
    self.safety.safety_rx_hook(self._send_interceptor_msg(0x0, 0x201))
    self.safety.set_gas_interceptor_detected(False)

  def test_disengage_on_gas(self):
    for long_controls_allowed in [0, 1]:
      self.safety.set_long_controls_allowed(long_controls_allowed)
      self.safety.safety_rx_hook(self._send_gas_msg(0))
      self.safety.set_controls_allowed(True)
      self.safety.safety_rx_hook(self._send_gas_msg(1))
      if long_controls_allowed:
        self.assertFalse(self.safety.get_controls_allowed())
      else:
        self.assertTrue(self.safety.get_controls_allowed())
    self.safety.set_long_controls_allowed(True)

  def test_allow_engage_with_gas_pressed(self):
    self.safety.safety_rx_hook(self._send_gas_msg(1))
    self.safety.set_controls_allowed(True)
    self.safety.safety_rx_hook(self._send_gas_msg(1))
    self.assertTrue(self.safety.get_controls_allowed())
    self.safety.safety_rx_hook(self._send_gas_msg(1))
    self.assertTrue(self.safety.get_controls_allowed())

  def test_disengage_on_gas_interceptor(self):
    for long_controls_allowed in [0, 1]:
      for g in range(0, 0x1000):
        self.safety.set_long_controls_allowed(long_controls_allowed)
        self.safety.safety_rx_hook(self._send_interceptor_msg(0, 0x201))
        self.safety.set_controls_allowed(True)
        self.safety.safety_rx_hook(self._send_interceptor_msg(g, 0x201))
        remain_enabled = (not long_controls_allowed or g <= INTERCEPTOR_THRESHOLD)
        self.assertEqual(remain_enabled, self.safety.get_controls_allowed())
        self.safety.safety_rx_hook(self._send_interceptor_msg(0, 0x201))
        self.safety.set_gas_interceptor_detected(False)
    self.safety.set_long_controls_allowed(True)

  def test_allow_engage_with_gas_interceptor_pressed(self):
    self.safety.safety_rx_hook(self._send_interceptor_msg(0x1000, 0x201))
    self.safety.set_controls_allowed(1)
    self.safety.safety_rx_hook(self._send_interceptor_msg(0x1000, 0x201))
    self.assertTrue(self.safety.get_controls_allowed())
    self.safety.safety_rx_hook(self._send_interceptor_msg(0, 0x201))
    self.safety.set_gas_interceptor_detected(False)

  def test_accel_actuation_limits(self):
    for long_controls_allowed in [0, 1]:
      self.safety.set_long_controls_allowed(long_controls_allowed)
      for accel in np.arange(MIN_ACCEL - 1000, MAX_ACCEL + 1000, 100):
        for controls_allowed in [True, False]:
          self.safety.set_controls_allowed(controls_allowed)
          if controls_allowed and long_controls_allowed:
            send = MIN_ACCEL <= accel <= MAX_ACCEL
          else:
            send = accel == 0
          self.assertEqual(send, self.safety.safety_tx_hook(self._accel_msg(accel)))
    self.safety.set_long_controls_allowed(True)

  def test_torque_absolute_limits(self):
    for controls_allowed in [True, False]:
      for torque in np.arange(-MAX_TORQUE - 1000, MAX_TORQUE + 1000, MAX_RATE_UP):
          self.safety.set_controls_allowed(controls_allowed)
          self.safety.set_toyota_rt_torque_last(torque)
          self.safety.set_toyota_torque_meas(torque, torque)
          self.safety.set_toyota_desired_torque_last(torque - MAX_RATE_UP)

          if controls_allowed:
            send = (-MAX_TORQUE <= torque <= MAX_TORQUE)
          else:
            send = torque == 0

          self.assertEqual(send, self.safety.safety_tx_hook(self._torque_msg(torque)))

  def test_non_realtime_limit_up(self):
    self.safety.set_controls_allowed(True)

    self._set_prev_torque(0)
    self.assertTrue(self.safety.safety_tx_hook(self._torque_msg(MAX_RATE_UP)))

    self._set_prev_torque(0)
    self.assertFalse(self.safety.safety_tx_hook(self._torque_msg(MAX_RATE_UP + 1)))

  def test_non_realtime_limit_down(self):
    self.safety.set_controls_allowed(True)

    self.safety.set_toyota_rt_torque_last(1000)
    self.safety.set_toyota_torque_meas(500, 500)
    self.safety.set_toyota_desired_torque_last(1000)
    self.assertTrue(self.safety.safety_tx_hook(self._torque_msg(1000 - MAX_RATE_DOWN)))

    self.safety.set_toyota_rt_torque_last(1000)
    self.safety.set_toyota_torque_meas(500, 500)
    self.safety.set_toyota_desired_torque_last(1000)
    self.assertFalse(self.safety.safety_tx_hook(self._torque_msg(1000 - MAX_RATE_DOWN + 1)))

  def test_exceed_torque_sensor(self):
    self.safety.set_controls_allowed(True)

    for sign in [-1, 1]:
      self._set_prev_torque(0)
      for t in np.arange(0, MAX_TORQUE_ERROR + 10, 10):
        t *= sign
        self.assertTrue(self.safety.safety_tx_hook(self._torque_msg(t)))

      self.assertFalse(self.safety.safety_tx_hook(self._torque_msg(sign * (MAX_TORQUE_ERROR + 10))))

  def test_realtime_limit_up(self):
    self.safety.set_controls_allowed(True)

    for sign in [-1, 1]:
      self.safety.init_tests_toyota()
      self._set_prev_torque(0)
      for t in np.arange(0, 380, 10):
        t *= sign
        self.safety.set_toyota_torque_meas(t, t)
        self.assertTrue(self.safety.safety_tx_hook(self._torque_msg(t)))
      self.assertFalse(self.safety.safety_tx_hook(self._torque_msg(sign * 380)))

      self._set_prev_torque(0)
      for t in np.arange(0, 370, 10):
        t *= sign
        self.safety.set_toyota_torque_meas(t, t)
        self.assertTrue(self.safety.safety_tx_hook(self._torque_msg(t)))

      # Increase timer to update rt_torque_last
      self.safety.set_timer(RT_INTERVAL + 1)
      self.assertTrue(self.safety.safety_tx_hook(self._torque_msg(sign * 370)))
      self.assertTrue(self.safety.safety_tx_hook(self._torque_msg(sign * 380)))

  def test_torque_measurements(self):
    self.safety.safety_rx_hook(self._torque_meas_msg(50))
    self.safety.safety_rx_hook(self._torque_meas_msg(-50))
    self.safety.safety_rx_hook(self._torque_meas_msg(0))
    self.safety.safety_rx_hook(self._torque_meas_msg(0))
    self.safety.safety_rx_hook(self._torque_meas_msg(0))
    self.safety.safety_rx_hook(self._torque_meas_msg(0))

    self.assertEqual(-51, self.safety.get_toyota_torque_meas_min())
    self.assertEqual(51, self.safety.get_toyota_torque_meas_max())

    self.safety.safety_rx_hook(self._torque_meas_msg(0))
    self.assertEqual(1, self.safety.get_toyota_torque_meas_max())
    self.assertEqual(-51, self.safety.get_toyota_torque_meas_min())

    self.safety.safety_rx_hook(self._torque_meas_msg(0))
    self.assertEqual(1, self.safety.get_toyota_torque_meas_max())
    self.assertEqual(-1, self.safety.get_toyota_torque_meas_min())

  def test_gas_interceptor_safety_check(self):

    self.safety.set_controls_allowed(0)
    self.assertTrue(self.safety.safety_tx_hook(self._send_interceptor_msg(0, 0x200)))
    self.assertFalse(self.safety.safety_tx_hook(self._send_interceptor_msg(0x1000, 0x200)))
    self.safety.set_controls_allowed(1)
    self.assertTrue(self.safety.safety_tx_hook(self._send_interceptor_msg(0x1000, 0x200)))

  def test_fwd_hook(self):

    buss = list(range(0x0, 0x3))
    msgs = list(range(0x1, 0x800))
    long_controls_allowed = [0, 1]
    toyota_camera_forwarded = [0, 1]

    for tcf in toyota_camera_forwarded:
      self.safety.set_toyota_camera_forwarded(tcf)
      for lca in long_controls_allowed:
        self.safety.set_long_controls_allowed(lca)
        blocked_msgs = [0x2E4, 0x412, 0x191]
        if lca:
          blocked_msgs += [0x343]
        for b in buss:
          for m in msgs:
            if tcf:
              if b == 0:
                fwd_bus = 2
              elif b == 1:
                fwd_bus = -1
              elif b == 2:
                fwd_bus = -1 if m in blocked_msgs else 0
            else:
              fwd_bus = -1

            # assume len 8
            self.assertEqual(fwd_bus, self.safety.safety_fwd_hook(b, self._send_msg(b, m, 8)))

    self.safety.set_long_controls_allowed(True)


if __name__ == "__main__":
  unittest.main()
