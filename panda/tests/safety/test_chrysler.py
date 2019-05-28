#!/usr/bin/env python2
import csv
import glob
import unittest
import numpy as np
import libpandasafety_py

MAX_RATE_UP = 3
MAX_RATE_DOWN = 3
MAX_STEER = 261

MAX_RT_DELTA = 112
RT_INTERVAL = 250000

MAX_TORQUE_ERROR = 80

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

def swap_bytes(data_str):
  """Accepts string with hex, returns integer with order swapped for CAN."""
  a = int(data_str, 16)
  return ((a & 0xff) << 24) + ((a & 0xff00) << 8) + ((a & 0x00ff0000) >> 8) + ((a & 0xff000000) >> 24)

class TestChryslerSafety(unittest.TestCase):
  @classmethod
  def setUp(cls):
    cls.safety = libpandasafety_py.libpandasafety
    cls.safety.nooutput_init(0)
    cls.safety.init_tests_chrysler()

  def _button_msg(self, buttons):
    to_send = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
    to_send[0].RIR = 1265 << 21
    to_send[0].RDLR = buttons
    return to_send

  def _set_prev_torque(self, t):
    self.safety.set_chrysler_desired_torque_last(t)
    self.safety.set_chrysler_rt_torque_last(t)
    self.safety.set_chrysler_torque_meas(t, t)

  def _torque_meas_msg(self, torque):
    to_send = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
    to_send[0].RIR = 544 << 21
    to_send[0].RDHR = ((torque + 1024) >> 8) + (((torque + 1024) & 0xff) << 8)
    return to_send

  def _torque_msg(self, torque):
    to_send = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
    to_send[0].RIR = 0x292 << 21
    to_send[0].RDLR = ((torque + 1024) >> 8) + (((torque + 1024) & 0xff) << 8)
    return to_send

  def test_default_controls_not_allowed(self):
    self.assertFalse(self.safety.get_controls_allowed())

  def test_steer_safety_check(self):
    for enabled in [0, 1]:
      for t in range(-MAX_STEER*2, MAX_STEER*2):
        self.safety.set_controls_allowed(enabled)
        self._set_prev_torque(t)
        if abs(t) > MAX_STEER or (not enabled and abs(t) > 0):
          self.assertFalse(self.safety.chrysler_tx_hook(self._torque_msg(t)))
        else:
          self.assertTrue(self.safety.chrysler_tx_hook(self._torque_msg(t)))

  def test_manually_enable_controls_allowed(self):
    self.safety.set_controls_allowed(1)
    self.assertTrue(self.safety.get_controls_allowed())
    self.safety.set_controls_allowed(0)
    self.assertFalse(self.safety.get_controls_allowed())

  def test_enable_control_allowed_from_cruise(self):
    to_push = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
    to_push[0].RIR = 0x1f4 << 21
    to_push[0].RDLR = 0x380000

    self.safety.chrysler_rx_hook(to_push)
    self.assertTrue(self.safety.get_controls_allowed())

  def test_disable_control_allowed_from_cruise(self):
    to_push = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
    to_push[0].RIR = 0x1f4 << 21
    to_push[0].RDLR = 0

    self.safety.set_controls_allowed(1)
    self.safety.chrysler_rx_hook(to_push)
    self.assertFalse(self.safety.get_controls_allowed())

  def test_non_realtime_limit_up(self):
    self.safety.set_controls_allowed(True)

    self._set_prev_torque(0)
    self.assertTrue(self.safety.chrysler_tx_hook(self._torque_msg(MAX_RATE_UP)))

    self._set_prev_torque(0)
    self.assertFalse(self.safety.chrysler_tx_hook(self._torque_msg(MAX_RATE_UP + 1)))

  def test_non_realtime_limit_down(self):
    self.safety.set_controls_allowed(True)

    self.safety.set_chrysler_rt_torque_last(MAX_STEER)
    torque_meas = MAX_STEER - MAX_TORQUE_ERROR - 20
    self.safety.set_chrysler_torque_meas(torque_meas, torque_meas)
    self.safety.set_chrysler_desired_torque_last(MAX_STEER)
    self.assertTrue(self.safety.chrysler_tx_hook(self._torque_msg(MAX_STEER - MAX_RATE_DOWN)))

    self.safety.set_chrysler_rt_torque_last(MAX_STEER)
    self.safety.set_chrysler_torque_meas(torque_meas, torque_meas)
    self.safety.set_chrysler_desired_torque_last(MAX_STEER)
    self.assertFalse(self.safety.chrysler_tx_hook(self._torque_msg(MAX_STEER - MAX_RATE_DOWN + 1)))

  def test_exceed_torque_sensor(self):
    self.safety.set_controls_allowed(True)

    for sign in [-1, 1]:
      self._set_prev_torque(0)
      for t in np.arange(0, MAX_TORQUE_ERROR + 2, 2):  # step needs to be smaller than MAX_TORQUE_ERROR
        t *= sign
        self.assertTrue(self.safety.chrysler_tx_hook(self._torque_msg(t)))

      self.assertFalse(self.safety.chrysler_tx_hook(self._torque_msg(sign * (MAX_TORQUE_ERROR + 2))))

  def test_realtime_limit_up(self):
    self.safety.set_controls_allowed(True)

    for sign in [-1, 1]:
      self.safety.init_tests_chrysler()
      self._set_prev_torque(0)
      for t in np.arange(0, MAX_RT_DELTA+1, 1):
        t *= sign
        self.safety.set_chrysler_torque_meas(t, t)
        self.assertTrue(self.safety.chrysler_tx_hook(self._torque_msg(t)))
      self.assertFalse(self.safety.chrysler_tx_hook(self._torque_msg(sign * (MAX_RT_DELTA + 1))))

      self._set_prev_torque(0)
      for t in np.arange(0, MAX_RT_DELTA+1, 1):
        t *= sign
        self.safety.set_chrysler_torque_meas(t, t)
        self.assertTrue(self.safety.chrysler_tx_hook(self._torque_msg(t)))

      # Increase timer to update rt_torque_last
      self.safety.set_timer(RT_INTERVAL + 1)
      self.assertTrue(self.safety.chrysler_tx_hook(self._torque_msg(sign * MAX_RT_DELTA)))
      self.assertTrue(self.safety.chrysler_tx_hook(self._torque_msg(sign * (MAX_RT_DELTA + 1))))

  def test_torque_measurements(self):
    self.safety.chrysler_rx_hook(self._torque_meas_msg(50))
    self.safety.chrysler_rx_hook(self._torque_meas_msg(-50))
    self.safety.chrysler_rx_hook(self._torque_meas_msg(0))
    self.safety.chrysler_rx_hook(self._torque_meas_msg(0))
    self.safety.chrysler_rx_hook(self._torque_meas_msg(0))
    self.safety.chrysler_rx_hook(self._torque_meas_msg(0))

    self.assertEqual(-50, self.safety.get_chrysler_torque_meas_min())
    self.assertEqual(50, self.safety.get_chrysler_torque_meas_max())

    self.safety.chrysler_rx_hook(self._torque_meas_msg(0))
    self.assertEqual(0, self.safety.get_chrysler_torque_meas_max())
    self.assertEqual(-50, self.safety.get_chrysler_torque_meas_min())

    self.safety.chrysler_rx_hook(self._torque_meas_msg(0))
    self.assertEqual(0, self.safety.get_chrysler_torque_meas_max())
    self.assertEqual(0, self.safety.get_chrysler_torque_meas_min())

  def _replay_drive(self, csv_reader):
    for row in csv_reader:
      if len(row) != 4:  # sometimes truncated at end of the file
        continue
      if row[0] == 'time':  # skip CSV header
        continue
      addr = int(row[1])
      bus = int(row[2])
      data_str = row[3]  # Example '081407ff0806e06f'
      to_send = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
      to_send[0].RIR = addr << 21
      to_send[0].RDHR = swap_bytes(data_str[8:])
      to_send[0].RDLR = swap_bytes(data_str[:8])
      if (bus == 128):
        self.assertTrue(self.safety.chrysler_tx_hook(to_send), msg=row)
      else:
        self.safety.chrysler_rx_hook(to_send)

  def test_replay_drive(self):
    # In Cabana, click "Save Log" and then put the downloaded CSV in this directory.
    test_files = glob.glob('chrysler_*.csv')
    for filename in test_files:
      print 'testing %s' % filename
      with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        self._replay_drive(reader)


if __name__ == "__main__":
  unittest.main()
