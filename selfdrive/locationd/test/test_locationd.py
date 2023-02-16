#!/usr/bin/env python3
import json
import random
import unittest
import time
import capnp

import cereal.messaging as messaging
from cereal.services import service_list
from common.params import Params
from common.transformations.coordinates import ecef2geodetic

from selfdrive.manager.process_config import managed_processes


class TestLocationdProc(unittest.TestCase):
  MAX_WAITS = 1000
  LLD_MSGS = ['gpsLocationExternal', 'cameraOdometry', 'carState', 'liveCalibration',
              'accelerometer', 'gyroscope', 'magnetometer']

  def setUp(self):
    random.seed(123489234)

    self.pm = messaging.PubMaster(self.LLD_MSGS)

    Params().put_bool("UbloxAvailable", True)
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

  def get_fake_msg(self, name, t):
    try:
      msg = messaging.new_message(name)
    except capnp.lib.capnp.KjException:
      msg = messaging.new_message(name, 0)


    if name == "gpsLocationExternal":
      msg.gpsLocationExternal.flags = 1
      msg.gpsLocationExternal.verticalAccuracy = 1.0
      msg.gpsLocationExternal.speedAccuracy = 1.0
      msg.gpsLocationExternal.bearingAccuracyDeg = 1.0
      msg.gpsLocationExternal.vNED = [0.0, 0.0, 0.0]
      msg.gpsLocationExternal.latitude = float(self.lat)
      msg.gpsLocationExternal.longitude = float(self.lon)
      msg.gpsLocationExternal.unixTimestampMillis = t * 1e6
      msg.gpsLocationExternal.altitude = float(self.alt)
    #if name == "gnssMeasurements":
    #  msg.gnssMeasurements.measTime = t
    #  msg.gnssMeasurements.positionECEF.value = [self.x , self.y, self.z]
    #  msg.gnssMeasurements.positionECEF.std = [0,0,0]
    #  msg.gnssMeasurements.positionECEF.valid = True
    #  msg.gnssMeasurements.velocityECEF.value = []
    #  msg.gnssMeasurements.velocityECEF.std = [0,0,0]
    #  msg.gnssMeasurements.velocityECEF.valid = True
    elif name == 'cameraOdometry':
      msg.cameraOdometry.rot = [0.0, 0.0, 0.0]
      msg.cameraOdometry.rotStd = [0.0, 0.0, 0.0]
      msg.cameraOdometry.trans = [0.0, 0.0, 0.0]
      msg.cameraOdometry.transStd = [0.0, 0.0, 0.0]
    msg.logMonoTime = t
    return msg

  def test_params_gps(self):
    # first reset params
    Params().remove('LastGPSPosition')

    self.x = -2710700 + (random.random() * 1e5)
    self.y = -4280600 + (random.random() * 1e5)
    self.z = 3850300 + (random.random() * 1e5)
    self.lat, self.lon, self.alt = ecef2geodetic([self.x, self.y, self.z])

    self.fake_duration = 90  # secs
    # get fake messages at the correct frequency, listed in services.py
    fake_msgs = []
    for sec in range(self.fake_duration):
      for name in self.LLD_MSGS:
        for j in range(int(service_list[name].frequency)):
          fake_msgs.append(self.get_fake_msg(name, int((sec + j / service_list[name].frequency) * 1e9)))

    for fake_msg in sorted(fake_msgs, key=lambda x: x.logMonoTime):
      self.send_msg(fake_msg)
    time.sleep(1)  # wait for async params write

    lastGPS = json.loads(Params().get('LastGPSPosition'))
    self.assertAlmostEqual(lastGPS['latitude'], self.lat, places=3)
    self.assertAlmostEqual(lastGPS['longitude'], self.lon, places=3)
    self.assertAlmostEqual(lastGPS['altitude'], self.alt, places=3)


if __name__ == "__main__":
  unittest.main()
