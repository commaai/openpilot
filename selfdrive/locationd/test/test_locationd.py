import pytest
import json
import random
import time
import capnp

import cereal.messaging as messaging
from cereal.services import SERVICE_LIST
from openpilot.common.params import Params
from openpilot.common.transformations.coordinates import ecef2geodetic

from openpilot.system.manager.process_config import managed_processes


class TestLocationdProc:
  LLD_MSGS = ['gpsLocationExternal', 'cameraOdometry', 'carState', 'liveCalibration',
              'accelerometer', 'gyroscope', 'magnetometer']

  def setup_method(self):
    self.pm = messaging.PubMaster(self.LLD_MSGS)

    self.params = Params()
    self.params.put_bool("UbloxAvailable", True)
    managed_processes['locationd'].prepare()
    managed_processes['locationd'].start()

  def teardown_method(self):
    managed_processes['locationd'].stop()

  def get_msg(self, name, t):
    try:
      msg = messaging.new_message(name)
    except capnp.lib.capnp.KjException:
      msg = messaging.new_message(name, 0)

    if name == "gpsLocationExternal":
      msg.gpsLocationExternal.flags = 1
      msg.gpsLocationExternal.hasFix = True
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
    msg.valid = True
    return msg

  def test_params_gps(self):
    random.seed(123489234)
    self.params.remove('LastGPSPosition')

    self.x = -2710700 + (random.random() * 1e5)
    self.y = -4280600 + (random.random() * 1e5)
    self.z = 3850300 + (random.random() * 1e5)
    self.lat, self.lon, self.alt = ecef2geodetic([self.x, self.y, self.z])

    # get fake messages at the correct frequency, listed in services.py
    msgs = []
    for sec in range(65):
      for name in self.LLD_MSGS:
        for j in range(int(SERVICE_LIST[name].frequency)):
          msgs.append(self.get_msg(name, int((sec + j / SERVICE_LIST[name].frequency) * 1e9)))

    for msg in sorted(msgs, key=lambda x: x.logMonoTime):
      self.pm.send(msg.which(), msg)
      if msg.which() == "cameraOdometry":
        self.pm.wait_for_readers_to_update(msg.which(), 0.1, dt=0.005)
    time.sleep(1)  # wait for async params write

    lastGPS = json.loads(self.params.get('LastGPSPosition'))
    assert lastGPS['latitude'] == pytest.approx(self.lat, abs=0.001)
    assert lastGPS['longitude'] == pytest.approx(self.lon, abs=0.001)
    assert lastGPS['altitude'] == pytest.approx(self.alt, abs=0.001)
