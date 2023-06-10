from collections import defaultdict

from cereal import messaging


def migrate_all(lr, old_logtime=False):
  msgs = migrate_sensorEvents(lr, old_logtime)
  msgs = migrate_carParams(msgs, old_logtime)
  msgs = migrate_cameraStates(msgs, old_logtime)

  return msgs


def migrate_cameraStates(lr, old_logtime=False):
  all_msgs = []
  min_frames_ids = {}
  prev_frame_ids = defaultdict(int)

  for msg in sorted(lr, key=lambda msg: msg.logMonoTime):
    if msg.which() not in ["roadCameraState", "wideRoadCameraState", "driverCameraState"]:
      all_msgs.append(msg)
      continue

    new_msg = messaging.new_message(msg.which())

    # old logs may have invalid frameIds - relative to route as a whole, not invidudual segment
    camera_state = getattr(msg, msg.which()).as_builder()
    assert camera_state.frameId >= prev_frame_ids[msg.which()], f"Corrupted camera state: {msg.which()}. Future camera state has lesser frameId."
    if msg.which() not in min_frames_ids:
      min_frames_ids[msg.which()] = camera_state.frameId
    camera_state.frameId = camera_state.frameId - min_frames_ids[msg.which()]
    prev_frame_ids[msg.which()] = camera_state.frameId

    if old_logtime:
      new_msg.logMonoTime = msg.logMonoTime
    new_msg.valid = msg.valid  
    setattr(new_msg, msg.which(), camera_state)

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
