#!/usr/bin/env python3
import bz2
import os
import time
import multiprocessing
import argparse
from tqdm import tqdm
# run DM procs
os.environ["USE_WEBCAM"] = "1"

import cereal.messaging as messaging
from cereal import car
from cereal.services import service_list
from cereal.visionipc import VisionIpcServer, VisionStreamType
from common.params import Params
from common.realtime import Ratekeeper, DT_MDL, DT_DMON, sec_since_boot
from common.transformations.camera import eon_f_frame_size, eon_d_frame_size, tici_f_frame_size, tici_d_frame_size, tici_e_frame_size
from panda.python import Panda
from selfdrive.car.toyota.values import EPS_SCALE
from selfdrive.manager.process import ensure_running
from selfdrive.manager.process_config import managed_processes
from selfdrive.test.process_replay.process_replay import CONFIGS, FAKEDATA, setup_env, check_openpilot_enabled
from selfdrive.test.update_ci_routes import upload_route
from tools.lib.route import Route
from tools.lib.framereader import FrameReader
from tools.lib.logreader import LogReader

def replay_panda_states(s, msgs):
  pm = messaging.PubMaster([s, 'peripheralState'])
  rk = Ratekeeper(service_list[s].frequency, print_delay_threshold=None)
  smsgs = [m for m in msgs if m.which() in ['pandaStates', 'pandaStateDEPRECATED']]

  # TODO: safety param migration should be handled automatically
  safety_param_migration = {
    "TOYOTA PRIUS 2017": EPS_SCALE["TOYOTA PRIUS 2017"] | Panda.FLAG_TOYOTA_STOCK_LONGITUDINAL,
    "TOYOTA RAV4 2017": EPS_SCALE["TOYOTA RAV4 2017"] | Panda.FLAG_TOYOTA_ALT_BRAKE,
    "KIA EV6 2022": Panda.FLAG_HYUNDAI_EV_GAS | Panda.FLAG_HYUNDAI_CANFD_HDA2,
  }

  # Migrate safety param base on carState
  cp = [m for m in msgs if m.which() == 'carParams'][0].carParams
  if cp.carFingerprint in safety_param_migration:
    safety_param = safety_param_migration[cp.carFingerprint]
  elif len(cp.safetyConfigs):
    safety_param = cp.safetyConfigs[0].safetyParam
    if cp.safetyConfigs[0].safetyParamDEPRECATED != 0:
      safety_param = cp.safetyConfigs[0].safetyParamDEPRECATED
  else:
    safety_param = cp.safetyParamDEPRECATED

  while True:
    for m in smsgs:
      if m.which() == 'pandaStateDEPRECATED':
        new_m = messaging.new_message('pandaStates', 1)
        new_m.pandaStates[0] = m.pandaStateDEPRECATED
        new_m.pandaStates[0].safetyParam = safety_param
        pm.send(s, new_m)
      else:
        new_m = m.as_builder()
        new_m.pandaStates[-1].safetyParam = safety_param
        new_m.logMonoTime = int(sec_since_boot() * 1e9)
      pm.send(s, new_m)

      new_m = messaging.new_message('peripheralState')
      pm.send('peripheralState', new_m)

      rk.keep_time()


def replay_manager_state(s, msgs):
  pm = messaging.PubMaster([s, ])
  rk = Ratekeeper(service_list[s].frequency, print_delay_threshold=None)

  while True:
    new_m = messaging.new_message('managerState')
    new_m.managerState.processes = [{'name': name, 'running': True} for name in managed_processes]
    pm.send(s, new_m)
    rk.keep_time()


def replay_device_state(s, msgs):
  pm = messaging.PubMaster([s, ])
  rk = Ratekeeper(service_list[s].frequency, print_delay_threshold=None)
  smsgs = [m for m in msgs if m.which() == s]
  while True:
    for m in smsgs:
      new_m = m.as_builder()
      new_m.logMonoTime = int(sec_since_boot() * 1e9)
      new_m.deviceState.freeSpacePercent = 50
      new_m.deviceState.memoryUsagePercent = 50
      pm.send(s, new_m)
      rk.keep_time()


def replay_sensor_event(s, msgs):
  pm = messaging.PubMaster([s, ])
  rk = Ratekeeper(service_list[s].frequency, print_delay_threshold=None)
  smsgs = [m for m in msgs if m.which() == s]
  while True:
    for m in smsgs:
      m = m.as_builder()
      m.logMonoTime = int(sec_since_boot() * 1e9)
      getattr(m, m.which()).timestamp = m.logMonoTime
      pm.send(m.which(), m)
      rk.keep_time()


def replay_service(s, msgs):
  pm = messaging.PubMaster([s, ])
  rk = Ratekeeper(service_list[s].frequency, print_delay_threshold=None)
  smsgs = [m for m in msgs if m.which() == s]
  while True:
    for m in smsgs:
      new_m = m.as_builder()
      new_m.logMonoTime = int(sec_since_boot() * 1e9)
      pm.send(s, new_m)
      rk.keep_time()


def replay_cameras(lr, frs, disable_tqdm=False):
  eon_cameras = [
    ("roadCameraState", DT_MDL, eon_f_frame_size, VisionStreamType.VISION_STREAM_ROAD, True),
    ("driverCameraState", DT_DMON, eon_d_frame_size, VisionStreamType.VISION_STREAM_DRIVER, False),
  ]
  tici_cameras = [
    ("roadCameraState", DT_MDL, tici_f_frame_size, VisionStreamType.VISION_STREAM_ROAD, False),
    ("wideRoadCameraState", DT_MDL, tici_e_frame_size, VisionStreamType.VISION_STREAM_WIDE_ROAD, False),
    ("driverCameraState", DT_DMON, tici_d_frame_size, VisionStreamType.VISION_STREAM_DRIVER, False),
  ]

  def replay_camera(s, stream, dt, vipc_server, frames, size, use_extra_client):
    services = [(s, stream)]
    if use_extra_client:
      services.append(("wideRoadCameraState", VisionStreamType.VISION_STREAM_WIDE_ROAD))
    pm = messaging.PubMaster([s for s, _ in services])
    rk = Ratekeeper(1 / dt, print_delay_threshold=None)

    img = b"\x00" * int(size[0] * size[1] * 3 / 2)
    while True:
      if frames is not None:
        img = frames[rk.frame % len(frames)]

      rk.keep_time()

      for s, stream in services:
        m = messaging.new_message(s)
        msg = getattr(m, s)
        msg.frameId = rk.frame
        msg.timestampSof = m.logMonoTime
        msg.timestampEof = m.logMonoTime
        pm.send(s, m)

        vipc_server.send(stream, img, msg.frameId, msg.timestampSof, msg.timestampEof)

  init_data = [m for m in lr if m.which() == 'initData'][0]
  cameras = tici_cameras if (init_data.initData.deviceType == 'tici') else eon_cameras

  # init vipc server and cameras
  p = []
  vs = VisionIpcServer("camerad")
  for (s, dt, size, stream, use_extra_client) in cameras:
    fr = frs.get(s, None)

    frames = None
    if fr is not None:
      print(f"Decompressing frames {s}")
      frames = []
      for i in tqdm(range(fr.frame_count), disable=disable_tqdm):
        img = fr.get(i, pix_fmt='nv12')[0]
        frames.append(img.flatten().tobytes())

    vs.create_buffers(stream, 40, False, size[0], size[1])
    if use_extra_client:
      vs.create_buffers(VisionStreamType.VISION_STREAM_WIDE_ROAD, 40, False, size[0], size[1])
    p.append(multiprocessing.Process(target=replay_camera,
                                     args=(s, stream, dt, vs, frames, size, use_extra_client)))

  vs.start_listener()
  return vs, p


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


def regen_segment(lr, frs=None, daemons="all", outdir=FAKEDATA, disable_tqdm=False):
  if not isinstance(daemons, str) and not hasattr(daemons, "__iter__"):
    raise ValueError("whitelist_proc must be a string or iterable")

  lr = migrate_carparams(list(lr))
  lr = migrate_sensorEvents(list(lr))
  if frs is None:
    frs = dict()

  # Get and setup initial state
  CP = [m for m in lr if m.which() == 'carParams'][0].carParams
  controlsState = [m for m in lr if m.which() == 'controlsState'][0].controlsState
  liveCalibration = [m for m in lr if m.which() == 'liveCalibration'][0]

  setup_env(CP=CP, controlsState=controlsState, log_dir=outdir)

  params = Params()
  params.put("CalibrationParams", liveCalibration.as_builder().to_bytes())

  vs, cam_procs = replay_cameras(lr, frs, disable_tqdm=disable_tqdm)
  fake_daemons = {
    'sensord': [
      multiprocessing.Process(target=replay_sensor_event, args=('accelerometer', lr)),
      multiprocessing.Process(target=replay_sensor_event, args=('gyroscope', lr)),
      multiprocessing.Process(target=replay_sensor_event, args=('magnetometer', lr)),
    ],
    'pandad': [
      multiprocessing.Process(target=replay_service, args=('can', lr)),
      multiprocessing.Process(target=replay_service, args=('ubloxRaw', lr)),
      multiprocessing.Process(target=replay_panda_states, args=('pandaStates', lr)),
    ],
    'manager': [
      multiprocessing.Process(target=replay_manager_state, args=('managerState', lr)),
    ],
    'thermald': [
      multiprocessing.Process(target=replay_device_state, args=('deviceState', lr)),
    ],
    'rawgpsd': [
      multiprocessing.Process(target=replay_service, args=('qcomGnss', lr)),
      multiprocessing.Process(target=replay_service, args=('gpsLocation', lr)),
    ],
    'camerad': [
      *cam_procs,
    ],
  }
  # TODO add configs for modeld, dmonitoringmodeld
  fakeable_daemons = {}
  for config in CONFIGS:
    processes = [
      multiprocessing.Process(target=replay_service, args=(msg, lr)) 
      for msg in config.subs
    ]
    fakeable_daemons[config.proc_name] = processes

  additional_fake_daemons = {}
  if daemons != "all":
    additional_fake_daemons = fakeable_daemons
    if isinstance(daemons, str):
      raise ValueError(f"Invalid value for daemons: {daemons}")

    for d in daemons:
      if d in fake_daemons:
        raise ValueError(f"Running daemon {d} is not supported!")
      
      if d in fakeable_daemons:
        del additional_fake_daemons[d]

  all_fake_daemons = {**fake_daemons, **additional_fake_daemons}

  try:
    # TODO: make first run of onnxruntime CUDA provider fast
    if "modeld" not in all_fake_daemons:
      managed_processes["modeld"].start()
    if "dmonitoringmodeld" not in all_fake_daemons:
      managed_processes["dmonitoringmodeld"].start()
    time.sleep(5)

    # start procs up
    ignore = list(all_fake_daemons.keys()) \
           + ['ui', 'manage_athenad', 'uploader', 'soundd', 'micd', 'navd']
    
    print("Faked daemons:", ", ".join(all_fake_daemons.keys()))
    print("Running daemons:", ", ".join([key for key in managed_processes.keys() if key not in ignore]))

    ensure_running(managed_processes.values(), started=True, params=Params(), CP=car.CarParams(), not_run=ignore)
    for procs in all_fake_daemons.values():
      for p in procs:
        p.start()

    for _ in tqdm(range(60), disable=disable_tqdm):
      # ensure all procs are running
      for d, procs in all_fake_daemons.items():
        for p in procs:
          if not p.is_alive():
            raise Exception(f"{d}'s {p.name} died")
      time.sleep(1)
  finally:
    # kill everything
    for p in managed_processes.values():
      p.stop()
    for procs in all_fake_daemons.values():
      for p in procs:
        p.terminate()

  del vs

  segment = params.get("CurrentRoute", encoding='utf-8') + "--0"
  seg_path = os.path.join(outdir, segment)
  # check to make sure openpilot is engaged in the route
  if not check_openpilot_enabled(LogReader(os.path.join(seg_path, "rlog"))):
    raise Exception(f"Route did not engage for long enough: {segment}")

  return seg_path


def regen_and_save(route, sidx, daemons="all", upload=False, use_route_meta=False, outdir=FAKEDATA, disable_tqdm=False):
  if use_route_meta:
    r = Route(route)
    lr = LogReader(r.log_paths()[sidx])
    fr = FrameReader(r.camera_paths()[sidx])
    if r.ecamera_paths()[sidx] is not None:
      wfr = FrameReader(r.ecamera_paths()[sidx])
    else:
      wfr = None
  else:
    lr = LogReader(f"cd:/{route.replace('|', '/')}/{sidx}/rlog.bz2")
    fr = FrameReader(f"cd:/{route.replace('|', '/')}/{sidx}/fcamera.hevc")
    device_type = next(iter(lr)).initData.deviceType
    if device_type == 'tici':
      wfr = FrameReader(f"cd:/{route.replace('|', '/')}/{sidx}/ecamera.hevc")
    else:
      wfr = None
  
  frs = {'roadCameraState': fr}
  if wfr is not None:
    frs['wideRoadCameraState'] = wfr
  rpath = regen_segment(lr, frs, daemons, outdir=outdir, disable_tqdm=disable_tqdm)

  # compress raw rlog before uploading
  with open(os.path.join(rpath, "rlog"), "rb") as f:
    data = bz2.compress(f.read())
  with open(os.path.join(rpath, "rlog.bz2"), "wb") as f:
    f.write(data)
  os.remove(os.path.join(rpath, "rlog"))

  lr = LogReader(os.path.join(rpath, 'rlog.bz2'))
  controls_state_active = [m.controlsState.active for m in lr if m.which() == 'controlsState']
  assert any(controls_state_active), "Segment did not engage"

  relr = os.path.relpath(rpath)

  print("\n\n", "*"*30, "\n\n")
  print("New route:", relr, "\n")
  if upload:
    upload_route(relr, exclude_patterns=['*.hevc', ])
  return relr


if __name__ == "__main__":
  def comma_separated_list(string):
    if string == "all":
      return string
    return string.split(",")

  parser = argparse.ArgumentParser(description="Generate new segments from old ones")
  parser.add_argument("--upload", action="store_true", help="Upload the new segment to the CI bucket")
  parser.add_argument("--outdir", help="log output dir", default=FAKEDATA)
  parser.add_argument("--whitelist-procs", type=comma_separated_list, default="all",
                      help="Comma-separated whitelist of processes to regen (e.g. controlsd). Pass 'all' to whitelist all processes.")
  parser.add_argument("route", type=str, help="The source route")
  parser.add_argument("seg", type=int, help="Segment in source route")
  args = parser.parse_args()

  regen_and_save(args.route, args.seg, args.whitelist_procs, args.upload, outdir=args.outdir)
