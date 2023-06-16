from collections import defaultdict

from cereal import messaging
from selfdrive.test.process_replay.vision_meta import meta_from_encode_index


def migrate_all(lr, old_logtime=False):
  msgs = migrate_sensorEvents(lr, old_logtime)
  msgs = migrate_carParams(msgs, old_logtime)
  msgs = migrate_cameraStates(msgs, old_logtime)

  return msgs


def migrate_cameraStates(lr, old_logtime=False):
  all_msgs = []
  prev_frame_ids = {}
  min_frame_logtimes = {}
  sorted_msgs = sorted(lr, key=lambda msg: msg.logMonoTime)
  
  for cam_state in ["roadCameraState", "wideRoadCameraState", "driverCameraState"]:
    min_frame_logtimes[cam_state] = next((msg.logMonoTime for msg in sorted_msgs if msg.which() == cam_state), None)

  for msg in sorted_msgs:
    if msg.which() not in ["roadCameraState", "wideRoadCameraState", "driverCameraState"]:
      all_msgs.append(msg)

    if msg.which() not in ["roadEncodeIdx", "wideRoadEncodeIdx", "driverEncodeIdx"]:
      continue

    meta = meta_from_encode_index(msg.which())
    new_msg = messaging.new_message(meta.camera_state)
    camera_state = getattr(new_msg, meta.camera_state)

    encode_index = getattr(msg, msg.which())
    assert encode_index.segmentId < 1200, f"Encoder index segmentId greater that 1200: {msg.which()} {encode_index.segmentId}"
    assert msg.which() not in prev_frame_ids or prev_frame_ids[msg.which()] + 1 == encode_index.segmentId, f"Corrupted encode index: {msg.which()}. FrameId is not sequential."
    camera_state.frameId = encode_index.segmentId
    camera_state.encodeId = encode_index.segmentId
    camera_state.timestampSof = encode_index.timestampSof
    camera_state.timestampEof = encode_index.timestampEof

    prev_frame_ids[msg.which()] = encode_index.segmentId

    new_msg.logMonoTime = min_frame_logtimes[meta.camera_state] + encode_index.segmentId * int(meta.dt * 1e9)
    new_msg.valid = msg.valid

    all_msgs.append(new_msg.as_reader())

  return all_msgs


def migrate_carParams(lr, old_logtime=False):
  all_msgs = []
  for msg in lr:
    if msg.which() == 'carParams':
      CP = messaging.new_message('carParams')
      CP.carParams = msg.carParams.as_builder()
      for car_fw in CP.carParams.carFw:
        car_fw.brand = CP.carParams.carName
      if old_logtime:
        CP.logMonoTime = msg.logMonoTime
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
