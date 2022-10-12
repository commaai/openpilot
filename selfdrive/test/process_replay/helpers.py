import os
import shutil
import uuid

import cereal.messaging as messaging
from common.params import Params

class OpenpilotPrefix(object):
  def __init__(self, prefix: str = None) -> None:
    self.prefix = prefix if prefix else str(uuid.uuid4())
    self.msgq_path = os.path.join('/dev/shm', self.prefix)

  def __enter__(self):
    os.environ['OPENPILOT_PREFIX'] = self.prefix
    try:
      os.mkdir(self.msgq_path)
    except FileExistsError:
      pass

  def __exit__(self, exc_type, exc_obj, exc_tb):
    symlink_path = Params().get_param_path()
    if os.path.exists(symlink_path):
      shutil.rmtree(os.path.realpath(symlink_path), ignore_errors=True)
      os.remove(symlink_path)
    shutil.rmtree(self.msgq_path, ignore_errors=True)
    del os.environ['OPENPILOT_PREFIX']
    return False


def migrate_carparams(lr):
  all_msgs = []
  for msg in lr:
    if msg.which() == 'carParams':
      CP = messaging.new_message('carParams')
      CP.carParams = msg.carParams.as_builder()
      for car_fw in CP.carParams.carFw:
        car_fw.brand = CP.carParams.carName
      msg = CP.as_reader()
    all_msgs.append(msg)

  return all_msgs


def migrate_sensorEvents(lr, old_logtime=False):
  all_msgs = []
  for msg in lr:
    if msg.which() != 'sensorEventsDEPRECATED':
      all_msgs.append(msg)
      continue

    # migrate to split sensor events
    for evt in msg.sensorEventsDEPRECATED:
      # build new message for each sensor type
      sensor_service = ''
      if evt.which() == 'acceleration':
        sensor_service = 'accelerometer'
      elif evt.which() == 'gyro' or evt.which() == 'gyroUncalibrated':
        sensor_service = 'gyroscope'
      elif evt.which() == 'light' or evt.which() == 'proximity':
        sensor_service = 'lightSensor'
      elif evt.which() == 'magnetic' or evt.which() == 'magneticUncalibrated':
        sensor_service = 'magnetometer'
      elif evt.which() == 'temperature':
        sensor_service = 'temperatureSensor'

      m = messaging.new_message(sensor_service)
      m.valid = True
      if old_logtime:
        m.logMonoTime = msg.logMonoTime

      m_dat = getattr(m, sensor_service)
      m_dat.version = evt.version
      m_dat.sensor = evt.sensor
      m_dat.type = evt.type
      m_dat.source = evt.source
      if old_logtime:
        m_dat.timestamp = evt.timestamp
      setattr(m_dat, evt.which(), getattr(evt, evt.which()))

      all_msgs.append(m.as_reader())

  return all_msgs
