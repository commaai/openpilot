#!/usr/bin/env python3
import unittest
import numpy as np
import libpandasafety_py  # pylint: disable=import-error
from panda import Panda

IPAS_OVERRIDE_THRESHOLD = 200

ANGLE_DELTA_BP = [0., 5., 15.]
ANGLE_DELTA_V = [5., .8, .15]     # windup limit
ANGLE_DELTA_VU = [5., 3.5, 0.4]   # unwind limit

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
    cls.safety.safety_set_mode(Panda.SAFETY_TOYOTA_IPAS, 66)
    cls.safety.init_tests_toyota()

  def _torque_driver_msg(self, torque):
    to_send = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
    to_send[0].RIR = 0x260 << 21

    t = twos_comp(torque, 16)
    to_send[0].RDLR = t | ((t & 0xFF) << 16)
    return to_send

  def _torque_driver_msg_array(self, torque):
    for i in range(6):
      self.safety.safety_rx_hook(self._torque_driver_msg(torque))

  def _angle_meas_msg(self, angle):
    to_send = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
    to_send[0].RIR = 0x25 << 21

    t = twos_comp(angle, 12)
    to_send[0].RDLR = ((t & 0xF00) >> 8) | ((t & 0xFF) << 8)
    return to_send

  def _angle_meas_msg_array(self, angle):
    for i in range(6):
      self.safety.safety_rx_hook(self._angle_meas_msg(angle))

  def _ipas_state_msg(self, state):
    to_send = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
    to_send[0].RIR = 0x262 << 21

    to_send[0].RDLR = state & 0xF
    return to_send

  def _ipas_control_msg(self, angle, state):
    # note: we command 2/3 of the angle due to CAN conversion
    to_send = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
    to_send[0].RIR = 0x266 << 21

    t = twos_comp(angle, 12)
    to_send[0].RDLR = ((t & 0xF00) >> 8) | ((t & 0xFF) << 8)
    to_send[0].RDLR |= ((state & 0xf) << 4)

    return to_send

  def _speed_msg(self, speed):
    to_send = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
    to_send[0].RIR = 0xb4 << 21
    speed = int(speed * 100 * 3.6)

    to_send[0].RDHR = ((speed & 0xFF) << 16) | (speed & 0xFF00)
    return to_send

  def test_ipas_override(self):

    ## angle control is not active
    self.safety.set_controls_allowed(1)

    # 3 consecutive msgs where driver exceeds threshold but angle_control isn't active
    self.safety.set_controls_allowed(1)
    self._torque_driver_msg_array(IPAS_OVERRIDE_THRESHOLD + 1)
    self.assertTrue(self.safety.get_controls_allowed())

    self._torque_driver_msg_array(-IPAS_OVERRIDE_THRESHOLD - 1)
    self.assertTrue(self.safety.get_controls_allowed())

    # ipas state is override
    self.safety.safety_rx_hook(self._ipas_state_msg(5))
    self.assertTrue(self.safety.get_controls_allowed())

    ## now angle control is active
    self.safety.safety_tx_hook(self._ipas_control_msg(0, 0))
    self.safety.safety_rx_hook(self._ipas_state_msg(0))

    # 3 consecutive msgs where driver does exceed threshold
    self.safety.set_controls_allowed(1)
    self._torque_driver_msg_array(IPAS_OVERRIDE_THRESHOLD + 1)
    self.assertFalse(self.safety.get_controls_allowed())

    self.safety.set_controls_allowed(1)
    self._torque_driver_msg_array(-IPAS_OVERRIDE_THRESHOLD - 1)
    self.assertFalse(self.safety.get_controls_allowed())

    # ipas state is override and torque isn't overriding any more
    self.safety.set_controls_allowed(1)
    self._torque_driver_msg_array(0)
    self.safety.safety_rx_hook(self._ipas_state_msg(5))
    self.assertFalse(self.safety.get_controls_allowed())

    # 3 consecutive msgs where driver does not exceed threshold and
    # ipas state is not override
    self.safety.set_controls_allowed(1)
    self.safety.safety_rx_hook(self._ipas_state_msg(0))
    self.assertTrue(self.safety.get_controls_allowed())

    self._torque_driver_msg_array(IPAS_OVERRIDE_THRESHOLD)
    self.assertTrue(self.safety.get_controls_allowed())

    self._torque_driver_msg_array(-IPAS_OVERRIDE_THRESHOLD)
    self.assertTrue(self.safety.get_controls_allowed())

    # reset no angle control at the end of the test
    self.safety.reset_angle_control()

  def test_angle_cmd_when_disabled(self):

    self.safety.set_controls_allowed(0)

    # test angle cmd too far from actual
    angle_refs = [-10, 10]
    deltas = list(range(-2, 3))
    expected_results = [False, True, True, True, False]

    for a in angle_refs:
      self._angle_meas_msg_array(a)
      for i, d in enumerate(deltas):
        self.assertEqual(expected_results[i], self.safety.safety_tx_hook(self._ipas_control_msg(a + d, 1)))

    # test ipas state cmd enabled
    self._angle_meas_msg_array(0)
    self.assertEqual(0, self.safety.safety_tx_hook(self._ipas_control_msg(0, 3)))

    # reset no angle control at the end of the test
    self.safety.reset_angle_control()

  def test_angle_cmd_when_enabled(self):

    # ipas angle cmd should pass through when controls are enabled

    self.safety.set_controls_allowed(1)
    self._angle_meas_msg_array(0)
    self.safety.safety_rx_hook(self._speed_msg(0.1))

    self.assertEqual(1, self.safety.safety_tx_hook(self._ipas_control_msg(0, 1)))
    self.assertEqual(1, self.safety.safety_tx_hook(self._ipas_control_msg(4, 1)))
    self.assertEqual(1, self.safety.safety_tx_hook(self._ipas_control_msg(0, 3)))
    self.assertEqual(1, self.safety.safety_tx_hook(self._ipas_control_msg(-4, 3)))
    self.assertEqual(1, self.safety.safety_tx_hook(self._ipas_control_msg(-8, 3)))

    # reset no angle control at the end of the test
    self.safety.reset_angle_control()

  def test_angle_cmd_rate_when_disabled(self):

    # as long as the command is close to the measured, no rate limit is enforced when
    # controls are disabled
    self.safety.set_controls_allowed(0)
    self.safety.safety_rx_hook(self._angle_meas_msg(0))
    self.assertEqual(1, self.safety.safety_tx_hook(self._ipas_control_msg(0, 1)))
    self.safety.safety_rx_hook(self._angle_meas_msg(100))
    self.assertEqual(1, self.safety.safety_tx_hook(self._ipas_control_msg(100, 1)))
    self.safety.safety_rx_hook(self._angle_meas_msg(-100))
    self.assertEqual(1, self.safety.safety_tx_hook(self._ipas_control_msg(-100, 1)))

    # reset no angle control at the end of the test
    self.safety.reset_angle_control()

  def test_angle_cmd_rate_when_enabled(self):

    # when controls are allowed, angle cmd rate limit is enforced
    # test 1: no limitations if we stay within limits
    speeds = [0., 1., 5., 10., 15., 100.]
    angles = [-300, -100, -10, 0, 10, 100, 300]
    for a in angles:
      for s in speeds:

        # first test against false positives
        self._angle_meas_msg_array(a)
        self.safety.safety_tx_hook(self._ipas_control_msg(a, 1))
        self.safety.set_controls_allowed(1)
        self.safety.safety_rx_hook(self._speed_msg(s))
        max_delta_up = int(np.interp(s, ANGLE_DELTA_BP, ANGLE_DELTA_V) * 2 / 3. + 1.)
        max_delta_down = int(np.interp(s, ANGLE_DELTA_BP, ANGLE_DELTA_VU) * 2 / 3. + 1.)
        self.assertEqual(True, self.safety.safety_tx_hook(self._ipas_control_msg(a + sign(a) * max_delta_up, 1)))
        self.assertTrue(self.safety.get_controls_allowed())
        self.assertEqual(True, self.safety.safety_tx_hook(self._ipas_control_msg(a, 1)))
        self.assertTrue(self.safety.get_controls_allowed())
        self.assertEqual(True, self.safety.safety_tx_hook(self._ipas_control_msg(a - sign(a) * max_delta_down, 1)))
        self.assertTrue(self.safety.get_controls_allowed())

        # now inject too high rates
        self.assertEqual(False, self.safety.safety_tx_hook(self._ipas_control_msg(a + sign(a) *
                                                                                  (max_delta_up + 1), 1)))
        self.assertFalse(self.safety.get_controls_allowed())
        self.safety.set_controls_allowed(1)
        self.assertEqual(True, self.safety.safety_tx_hook(self._ipas_control_msg(a + sign(a) * max_delta_up, 1)))
        self.assertTrue(self.safety.get_controls_allowed())
        self.assertEqual(True, self.safety.safety_tx_hook(self._ipas_control_msg(a, 1)))
        self.assertTrue(self.safety.get_controls_allowed())
        self.assertEqual(False, self.safety.safety_tx_hook(self._ipas_control_msg(a - sign(a) *
                                                                                  (max_delta_down + 1), 1)))
        self.assertFalse(self.safety.get_controls_allowed())

    # reset no angle control at the end of the test
    self.safety.reset_angle_control()

  def test_angle_measured_rate(self):

    speeds = [0., 1., 5., 10., 15., 100.]
    angles = [-300, -100, -10, 0, 10, 100, 300]
    angles = [10]
    for a in angles:
      for s in speeds:
        self._angle_meas_msg_array(a)
        self.safety.safety_tx_hook(self._ipas_control_msg(a, 1))
        self.safety.set_controls_allowed(1)
        self.safety.safety_rx_hook(self._speed_msg(s))
        #max_delta_up = int(np.interp(s, ANGLE_DELTA_BP, ANGLE_DELTA_V) * 2 / 3. + 1.)
        #max_delta_down = int(np.interp(s, ANGLE_DELTA_BP, ANGLE_DELTA_VU) * 2 / 3. + 1.)
        self.safety.safety_rx_hook(self._angle_meas_msg(a))
        self.assertTrue(self.safety.get_controls_allowed())
        self.safety.safety_rx_hook(self._angle_meas_msg(a + 150))
        self.assertFalse(self.safety.get_controls_allowed())

    # reset no angle control at the end of the test
    self.safety.reset_angle_control()


if __name__ == "__main__":
  unittest.main()
