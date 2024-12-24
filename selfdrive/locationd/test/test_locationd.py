import capnp

import cereal.messaging as messaging
from openpilot.common.params import Params

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
