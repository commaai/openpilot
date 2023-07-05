#!/usr/bin/env python3
"""This test can't be run together with other locationd tests.
cffi.dlopen breaks the list of registered filters."""
import os
import random
import unittest

from cffi import FFI

import cereal.messaging as messaging
from cereal import log

from common.ffi_wrapper import suffix

SENSOR_DECIMATION = 1
VISION_DECIMATION = 1

LIBLOCATIOND_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../liblocationd' + suffix()))


class TestLocationdLib(unittest.TestCase):
  def setUp(self):
    header = '''typedef ...* Localizer_t;
Localizer_t localizer_init(bool has_ublox);
void localizer_get_message_bytes(Localizer_t localizer, bool inputsOK, bool sensorsOK, bool gpsOK, bool msgValid, char *buff, size_t buff_size);
void localizer_handle_msg_bytes(Localizer_t localizer, const char *data, size_t size);'''

    self.ffi = FFI()
    self.ffi.cdef(header)
    self.lib = self.ffi.dlopen(LIBLOCATIOND_PATH)

    self.localizer = self.lib.localizer_init(True) # default to ublox

    self.buff_size = 2048
    self.msg_buff = self.ffi.new(f'char[{self.buff_size}]')

  def localizer_handle_msg(self, msg_builder):
    bytstr = msg_builder.to_bytes()
    self.lib.localizer_handle_msg_bytes(self.localizer, self.ffi.from_buffer(bytstr), len(bytstr))

  def localizer_get_msg(self, t=0, inputsOK=True, sensorsOK=True, gpsOK=True, msgValid=True):
    self.lib.localizer_get_message_bytes(self.localizer, inputsOK, sensorsOK, gpsOK, msgValid, self.ffi.addressof(self.msg_buff, 0), self.buff_size)
    return log.Event.from_bytes(self.ffi.buffer(self.msg_buff), nesting_limit=self.buff_size // 8)

  def test_liblocalizer(self):
    msg = messaging.new_message('liveCalibration')
    msg.liveCalibration.validBlocks = random.randint(1, 10)
    msg.liveCalibration.rpyCalib = [random.random() / 10 for _ in range(3)]

    self.localizer_handle_msg(msg)
    liveloc = self.localizer_get_msg()
    self.assertTrue(liveloc is not None)

  @unittest.skip("temporarily disabled due to false positives")
  def test_device_fell(self):
    msg = messaging.new_message('accelerometer')
    msg.accelerometer.sensor = 1
    msg.accelerometer.timestamp = msg.logMonoTime
    msg.accelerometer.type = 1
    msg.accelerometer.init('acceleration')
    msg.accelerometer.acceleration.v = [10.0, 0.0, 0.0]  # zero with gravity
    self.localizer_handle_msg(msg)

    ret = self.localizer_get_msg()
    self.assertTrue(ret.liveLocationKalman.deviceStable)

    msg = messaging.new_message('accelerometer')
    msg.accelerometer.sensor = 1
    msg.accelerometer.timestamp = msg.logMonoTime
    msg.accelerometer.type = 1
    msg.accelerometer.init('acceleration')
    msg.accelerometer.acceleration.v = [50.1, 0.0, 0.0]  # more than 40 m/s**2
    self.localizer_handle_msg(msg)

    ret = self.localizer_get_msg()
    self.assertFalse(ret.liveLocationKalman.deviceStable)

  def test_posenet_spike(self):
    for _ in range(SENSOR_DECIMATION):
      msg = messaging.new_message('carState')
      msg.carState.vEgo = 6.0  # more than 5 m/s
      self.localizer_handle_msg(msg)

    ret = self.localizer_get_msg()
    self.assertTrue(ret.liveLocationKalman.posenetOK)

    for _ in range(20 * VISION_DECIMATION):  # size of hist_old
      msg = messaging.new_message('cameraOdometry')
      msg.cameraOdometry.rot = [0.0, 0.0, 0.0]
      msg.cameraOdometry.rotStd = [0.1, 0.1, 0.1]
      msg.cameraOdometry.trans = [0.0, 0.0, 0.0]
      msg.cameraOdometry.transStd = [2.0, 0.1, 0.1]
      self.localizer_handle_msg(msg)

    for _ in range(20 * VISION_DECIMATION):  # size of hist_new
      msg = messaging.new_message('cameraOdometry')
      msg.cameraOdometry.rot = [0.0, 0.0, 0.0]
      msg.cameraOdometry.rotStd = [1.0, 1.0, 1.0]
      msg.cameraOdometry.trans = [0.0, 0.0, 0.0]
      msg.cameraOdometry.transStd = [10.1, 0.1, 0.1]  # more than 4 times larger
      self.localizer_handle_msg(msg)

    ret = self.localizer_get_msg()
    self.assertFalse(ret.liveLocationKalman.posenetOK)

if __name__ == "__main__":
  unittest.main()

