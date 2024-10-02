from collections import defaultdict

from cereal import messaging, car
from opendbc.car.fingerprints import MIGRATION
from opendbc.car.toyota.values import EPS_SCALE
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.fill_model_msg import fill_xyz_poly, fill_lane_line_meta
from openpilot.selfdrive.test.process_replay.vision_meta import meta_from_encode_index
from openpilot.system.manager.process_config import managed_processes
from panda import Panda


# TODO: message migration should happen in-place
def migrate_all(lr, manager_states=False, panda_states=False, camera_states=False):
  msgs = migrate_sensorEvents(lr)
  msgs = migrate_carParams(msgs)
  msgs = migrate_gpsLocation(msgs)
  msgs = migrate_deviceState(msgs)
  msgs = migrate_carOutput(msgs)
  msgs = migrate_controlsState(msgs)
  msgs = migrate_liveLocationKalman(msgs)
  msgs = migrate_liveTracks(msgs)
  msgs = migrate_driverAssistance(msgs)
  msgs = migrate_drivingModelData(msgs)
  if manager_states:
    msgs = migrate_managerState(msgs)
  if panda_states:
    msgs = migrate_pandaStates(msgs)
    msgs = migrate_peripheralState(msgs)
  if camera_states:
    msgs = migrate_cameraStates(msgs)

  return msgs


def migrate_driverAssistance(lr):
  all_msgs = []
  for msg in lr:
    all_msgs.append(msg)
    if msg.which() == 'longitudinalPlan':
      all_msgs.append(messaging.new_message('driverAssistance', valid=True, logMonoTime=msg.logMonoTime).as_reader())
    if msg.which() == 'driverAssistance':
      return lr
  return all_msgs


def migrate_drivingModelData(lr):
  all_msgs = []
  for msg in lr:
    all_msgs.append(msg)
    if msg.which() == "modelV2":
      dmd = messaging.new_message('drivingModelData', valid=msg.valid, logMonoTime=msg.logMonoTime)
      for field in ["frameId", "frameIdExtra", "frameDropPerc", "modelExecutionTime", "action"]:
        setattr(dmd.drivingModelData, field, getattr(msg.modelV2, field))
      for meta_field in ["laneChangeState", "laneChangeState"]:
        setattr(dmd.drivingModelData.meta, meta_field, getattr(msg.modelV2.meta, meta_field))
      if len(msg.modelV2.laneLines) and len(msg.modelV2.laneLineProbs):
        fill_lane_line_meta(dmd.drivingModelData.laneLineMeta, msg.modelV2.laneLines, msg.modelV2.laneLineProbs)
      if all(len(a) for a in [msg.modelV2.position.x, msg.modelV2.position.y, msg.modelV2.position.z]):
        fill_xyz_poly(dmd.drivingModelData.path, ModelConstants.POLY_PATH_DEGREE, msg.modelV2.position.x, msg.modelV2.position.y, msg.modelV2.position.z)
      all_msgs.append(dmd.as_reader())
    elif msg.which() == "drivingModelData":
      return lr
  return all_msgs


def migrate_liveTracks(lr):
  all_msgs = []
  for msg in lr:
    if msg.which() != "liveTracksDEPRECATED":
      all_msgs.append(msg)
      continue

    new_msg = messaging.new_message('liveTracks')
    new_msg.valid = msg.valid
    new_msg.logMonoTime = msg.logMonoTime

    pts = []
    for track in msg.liveTracksDEPRECATED:
      pt = car.RadarData.RadarPoint()
      pt.trackId = track.trackId

      pt.dRel = track.dRel
      pt.yRel = track.yRel
      pt.vRel = track.vRel
      pt.aRel = track.aRel
      pt.measured = True
      pts.append(pt)

    new_msg.liveTracks.points = pts
    all_msgs.append(new_msg.as_reader())

  return all_msgs


def migrate_liveLocationKalman(lr):
  # migration needed only for routes before livePose
  if any(msg.which() == 'livePose' for msg in lr):
    return lr

  all_msgs = []
  for msg in lr:
    if msg.which() != 'liveLocationKalmanDEPRECATED':
      all_msgs.append(msg)
      continue

    m = messaging.new_message('livePose')
    m.valid = msg.valid
    m.logMonoTime = msg.logMonoTime
    for field in ["orientationNED", "velocityDevice", "accelerationDevice", "angularVelocityDevice"]:
      lp_field, llk_field = getattr(m.livePose, field), getattr(msg.liveLocationKalmanDEPRECATED, field)
      lp_field.x, lp_field.y, lp_field.z = llk_field.value
      lp_field.xStd, lp_field.yStd, lp_field.zStd = llk_field.std
      lp_field.valid = llk_field.valid
    for flag in ["inputsOK", "posenetOK", "sensorsOK"]:
      setattr(m.livePose, flag, getattr(msg.liveLocationKalmanDEPRECATED, flag))

    all_msgs.append(m.as_reader())

  return all_msgs


def migrate_controlsState(lr):
  ret = []
  last_cs = None
  for msg in lr:
    if msg.which() == 'controlsState':
      last_cs = msg

      m = messaging.new_message('selfdriveState')
      m.valid = msg.valid
      m.logMonoTime = msg.logMonoTime
      ss = m.selfdriveState
      for field in ("enabled", "active", "state", "engageable", "alertText1", "alertText2",
                    "alertStatus", "alertSize", "alertType", "experimentalMode",
                    "personality"):
        setattr(ss, field, getattr(msg.controlsState, field+"DEPRECATED"))
      ret.append(m.as_reader())
    elif msg.which() == 'carState' and last_cs is not None:
      if last_cs.controlsState.vCruiseDEPRECATED - msg.carState.vCruise > 0.1:
        msg = msg.as_builder()
        msg.carState.vCruise = last_cs.controlsState.vCruiseDEPRECATED
        msg.carState.vCruiseCluster = last_cs.controlsState.vCruiseClusterDEPRECATED
        msg = msg.as_reader()

    ret.append(msg)
  return ret


def migrate_managerState(lr):
  all_msgs = []
  for msg in lr:
    if msg.which() != "managerState":
      all_msgs.append(msg)
      continue

    new_msg = msg.as_builder()
    new_msg.managerState.processes = [{'name': name, 'running': True} for name in managed_processes]
    all_msgs.append(new_msg.as_reader())

  return all_msgs


def migrate_gpsLocation(lr):
  all_msgs = []
  for msg in lr:
    if msg.which() in ('gpsLocation', 'gpsLocationExternal'):
      new_msg = msg.as_builder()
      g = getattr(new_msg, new_msg.which())
      # hasFix is a newer field
      if not g.hasFix and g.flags == 1:
        g.hasFix = True
      all_msgs.append(new_msg.as_reader())
    else:
      all_msgs.append(msg)
  return all_msgs


def migrate_deviceState(lr):
  all_msgs = []
  dt = None
  for msg in lr:
    if msg.which() == 'initData':
      dt = msg.initData.deviceType
    if msg.which() == 'deviceState':
      n = msg.as_builder()
      n.deviceState.deviceType = dt
      all_msgs.append(n.as_reader())
    else:
      all_msgs.append(msg)
  return all_msgs


def migrate_carOutput(lr):
  # migration needed only for routes before carOutput
  if any(msg.which() == 'carOutput' for msg in lr):
    return lr

  all_msgs = []
  for msg in lr:
    if msg.which() == 'carControl':
      co = messaging.new_message('carOutput')
      co.valid = msg.valid
      co.logMonoTime = msg.logMonoTime
      co.carOutput.actuatorsOutput = msg.carControl.actuatorsOutputDEPRECATED
      all_msgs.append(co.as_reader())
    all_msgs.append(msg)
  return all_msgs


def migrate_pandaStates(lr):
  all_msgs = []
  # TODO: safety param migration should be handled automatically
  safety_param_migration = {
    "TOYOTA_PRIUS": EPS_SCALE["TOYOTA_PRIUS"] | Panda.FLAG_TOYOTA_STOCK_LONGITUDINAL,
    "TOYOTA_RAV4": EPS_SCALE["TOYOTA_RAV4"] | Panda.FLAG_TOYOTA_ALT_BRAKE,
    "KIA_EV6": Panda.FLAG_HYUNDAI_EV_GAS | Panda.FLAG_HYUNDAI_CANFD_HDA2,
  }

  # Migrate safety param base on carState
  CP = next((m.carParams for m in lr if m.which() == 'carParams'), None)
  assert CP is not None, "carParams message not found"
  if CP.carFingerprint in safety_param_migration:
    safety_param = safety_param_migration[CP.carFingerprint]
  elif len(CP.safetyConfigs):
    safety_param = CP.safetyConfigs[0].safetyParam
    if CP.safetyConfigs[0].safetyParamDEPRECATED != 0:
      safety_param = CP.safetyConfigs[0].safetyParamDEPRECATED
  else:
    safety_param = CP.safetyParamDEPRECATED

  for msg in lr:
    if msg.which() == 'pandaStateDEPRECATED':
      new_msg = messaging.new_message('pandaStates', 1)
      new_msg.valid = msg.valid
      new_msg.logMonoTime = msg.logMonoTime
      new_msg.pandaStates[0] = msg.pandaStateDEPRECATED
      new_msg.pandaStates[0].safetyParam = safety_param
      all_msgs.append(new_msg.as_reader())
    elif msg.which() == 'pandaStates':
      new_msg = msg.as_builder()
      new_msg.pandaStates[-1].safetyParam = safety_param
      all_msgs.append(new_msg.as_reader())
    else:
      all_msgs.append(msg)

  return all_msgs


def migrate_peripheralState(lr):
  if any(msg.which() == "peripheralState" for msg in lr):
    return lr

  all_msg = []
  for msg in lr:
    all_msg.append(msg)
    if msg.which() not in ["pandaStates", "pandaStateDEPRECATED"]:
      continue

    new_msg = messaging.new_message("peripheralState")
    new_msg.valid = msg.valid
    new_msg.logMonoTime = msg.logMonoTime
    all_msg.append(new_msg.as_reader())

  return all_msg


def migrate_cameraStates(lr):
  all_msgs = []
  frame_to_encode_id = defaultdict(dict)
  # just for encodeId fallback mechanism
  min_frame_id = defaultdict(lambda: float('inf'))

  for msg in lr:
    if msg.which() not in ["roadEncodeIdx", "wideRoadEncodeIdx", "driverEncodeIdx"]:
      continue

    encode_index = getattr(msg, msg.which())
    meta = meta_from_encode_index(msg.which())

    assert encode_index.segmentId < 1200, f"Encoder index segmentId greater that 1200: {msg.which()} {encode_index.segmentId}"
    frame_to_encode_id[meta.camera_state][encode_index.frameId] = encode_index.segmentId

  for msg in lr:
    if msg.which() not in ["roadCameraState", "wideRoadCameraState", "driverCameraState"]:
      all_msgs.append(msg)
      continue

    camera_state = getattr(msg, msg.which())
    min_frame_id[msg.which()] = min(min_frame_id[msg.which()], camera_state.frameId)

    encode_id = frame_to_encode_id[msg.which()].get(camera_state.frameId)
    if encode_id is None:
      print(f"Missing encoded frame for camera feed {msg.which()} with frameId: {camera_state.frameId}")
      if len(frame_to_encode_id[msg.which()]) != 0:
        continue

      # fallback mechanism for logs without encodeIdx (e.g. logs from before 2022 with dcamera recording disabled)
      # try to fake encode_id by subtracting lowest frameId
      encode_id = camera_state.frameId - min_frame_id[msg.which()]
      print(f"Faking encodeId to {encode_id} for camera feed {msg.which()} with frameId: {camera_state.frameId}")

    new_msg = messaging.new_message(msg.which())
    new_camera_state = getattr(new_msg, new_msg.which())
    new_camera_state.frameId = encode_id
    new_camera_state.encodeId = encode_id
    # timestampSof was added later so it might be missing on some old segments
    if camera_state.timestampSof == 0 and camera_state.timestampEof > 25000000:
      new_camera_state.timestampSof = camera_state.timestampEof - 18000000
    else:
      new_camera_state.timestampSof = camera_state.timestampSof
    new_camera_state.timestampEof = camera_state.timestampEof
    new_msg.logMonoTime = msg.logMonoTime
    new_msg.valid = msg.valid

    all_msgs.append(new_msg.as_reader())

  return all_msgs


def migrate_carParams(lr):
  all_msgs = []
  for msg in lr:
    if msg.which() == 'carParams':
      CP = msg.as_builder()
      CP.carParams.carFingerprint = MIGRATION.get(CP.carParams.carFingerprint, CP.carParams.carFingerprint)
      for car_fw in CP.carParams.carFw:
        car_fw.brand = CP.carParams.carName
      CP.logMonoTime = msg.logMonoTime
      msg = CP.as_reader()
    all_msgs.append(msg)

  return all_msgs


def migrate_sensorEvents(lr):
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
      m.logMonoTime = msg.logMonoTime

      m_dat = getattr(m, sensor_service)
      m_dat.version = evt.version
      m_dat.sensor = evt.sensor
      m_dat.type = evt.type
      m_dat.source = evt.source
      m_dat.timestamp = evt.timestamp
      setattr(m_dat, evt.which(), getattr(evt, evt.which()))

      all_msgs.append(m.as_reader())

  return all_msgs
