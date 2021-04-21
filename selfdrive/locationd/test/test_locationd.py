#!/usr/bin/env python3
import json
import random
import unittest
import time
from cffi import FFI

from cereal import log
import cereal.messaging as messaging
from common.params import Params

from selfdrive.manager.process_config import managed_processes

random.seed(123489234)


class TestLocationdLib(unittest.TestCase):
  def setUp(self):
    header = '''typedef ...* Localizer_t;
Localizer_t localizer_init();
void localizer_get_message_bytes(Localizer_t localizer, uint64_t logMonoTime, bool inputsOK, bool sensorsOK, bool gpsOK, char *buff, size_t buff_size);
void localizer_handle_msg_bytes(Localizer_t localizer, const char *data, size_t size);'''
    localizer_bin = '/home/batman/openpilot/selfdrive/locationd/liblocationd.so'

    self.ffi = FFI()
    self.ffi.cdef(header)
    self.lib = self.ffi.dlopen(localizer_bin)

    self.localizer = self.lib.localizer_init()

    self.buff_size = 2048
    self.msg_buff = self.ffi.new(f'char[{self.buff_size}]')

  def localizer_handle_msg(self, msg_builder):
    bytstr = msg_builder.to_bytes()
    self.lib.localizer_handle_msg_bytes(self.localizer, self.ffi.from_buffer(bytstr), len(bytstr))

  def localizer_get_msg_dict(self, t=0, inputsOK=True, sensorsOK=True, gpsOK=True):
    self.lib.localizer_get_message_bytes(self.localizer, t, inputsOK, sensorsOK, gpsOK, self.ffi.addressof(self.msg_buff, 0), self.buff_size)
    return log.Event.from_bytes(self.ffi.buffer(self.msg_buff), nesting_limit=self.buff_size // 8).to_dict()

  def test_liblocalizer(self):
    msg = messaging.new_message('liveCalibration')
    msg.liveCalibration.validBlocks = random.randint(1, 10)
    msg.liveCalibration.rpyCalib = [random.random() for _ in range(3)]

    self.localizer_handle_msg(msg)
    liveloc = self.localizer_get_msg_dict()
    self.assertTrue(liveloc is not None)

  def test_device_fell(self):
    msg = messaging.new_message('sensorEvents', 1)
    msg.sensorEvents[0].sensor = 4
    msg.sensorEvents[0].type = 0x10
    msg.sensorEvents[0].init('acceleration')
    msg.sensorEvents[0].acceleration.v = [0.0, 0.0, 0.0]
    self.localizer_handle_msg(msg)
    self.localizer_get_msg_dict()

  def test_gyro_uncalibrated(self):
    msg = messaging.new_message('sensorEvents', 1)
    msg.sensorEvents[0].sensor = 5
    msg.sensorEvents[0].type = 0x10
    msg.sensorEvents[0].init('gyroUncalibrated')
    msg.sensorEvents[0].gyroUncalibrated.v = [0.0, 0.0, 0.0]
    self.localizer_handle_msg(msg)
    self.localizer_get_msg_dict()


class TestLocationdProc(unittest.TestCase):
  MAX_WAITS = 1000

  def setUp(self):
    self.pm = messaging.PubMaster({'gpsLocationExternal', 'cameraOdometry'})
    # sockets = {s: messaging.sub_sock(s, timeout=1000) for s in sub_sockets}

    managed_processes['locationd'].prepare()
    managed_processes['locationd'].start()

    time.sleep(1)

  def tearDown(self):
    managed_processes['locationd'].stop()

  def send_msg(self, msg):
    self.pm.send(msg.which(), msg)
    waits_left = self.MAX_WAITS
    while waits_left and not self.pm.all_readers_updated(msg.which()):
      time.sleep(0)
      waits_left -= 1
    time.sleep(0.0001)

  def test_params_gps(self):
    # first reset params
    Params().put('LastGPSPosition', json.dumps({"latitude": 0.0, "longitude": 0.0, "altitude": 0.0}))

    lat = 30 + (random.random() * 10.0)
    lon = -70 + (random.random() * 10.0)
    alt = 5 + (random.random() * 10.0)

    for _ in range(1000):  # because of kalman filter, send often
      msg = messaging.new_message('gpsLocationExternal')
      msg.logMonoTime = 0
      msg.gpsLocationExternal.flags = 1
      msg.gpsLocationExternal.verticalAccuracy = 0.0
      msg.gpsLocationExternal.vNED = [0.0, 0.0, 0.0]
      msg.gpsLocationExternal.latitude = lat
      msg.gpsLocationExternal.longitude = lon
      msg.gpsLocationExternal.altitude = alt
      self.send_msg(msg)

    for _ in range(250):  # params is only written so often
      msg = messaging.new_message('cameraOdometry')
      msg.logMonoTime = 0
      msg.cameraOdometry.frameId = 1200
      msg.cameraOdometry.rot = [0.0, 0.0, 0.0]
      msg.cameraOdometry.rotStd = [0.0, 0.0, 0.0]
      msg.cameraOdometry.trans = [0.0, 0.0, 0.0]
      msg.cameraOdometry.transStd = [0.0, 0.0, 0.0]
      self.send_msg(msg)

    time.sleep(1)  # wait for async params write

    lastGPS = json.loads(Params().get('LastGPSPosition'))

    self.assertAlmostEqual(lastGPS['latitude'], lat, places=3)
    self.assertAlmostEqual(lastGPS['longitude'], lon, places=3)
    self.assertAlmostEqual(lastGPS['altitude'], alt, places=3)


if __name__ == "__main__":
  unittest.main()
