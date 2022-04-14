#!/usr/bin/env python3
import os
import time
import multiprocessing
from tqdm import tqdm
import argparse
# run DM procs
os.environ["USE_WEBCAM"] = "1"

import cereal.messaging as messaging
from cereal.services import service_list
from cereal.visionipc.visionipc_pyx import VisionIpcServer, VisionStreamType  # pylint: disable=no-name-in-module, import-error
from common.params import Params
from common.realtime import Ratekeeper, DT_MDL, DT_DMON, sec_since_boot
from common.transformations.camera import eon_f_frame_size, eon_d_frame_size, tici_f_frame_size, tici_d_frame_size
from selfdrive.car.fingerprints import FW_VERSIONS
from selfdrive.manager.process import ensure_running
from selfdrive.manager.process_config import managed_processes
from selfdrive.test.process_replay.process_replay import setup_env, check_enabled
from selfdrive.test.update_ci_routes import upload_route
from tools.lib.route import Route
from tools.lib.framereader import FrameReader
from tools.lib.logreader import LogReader


process_replay_dir = os.path.dirname(os.path.abspath(__file__))
FAKEDATA = os.path.join(process_replay_dir, "fakedata/")


def replay_panda_states(s, msgs):
  pm = messaging.PubMaster([s, 'peripheralState'])
  rk = Ratekeeper(service_list[s].frequency, print_delay_threshold=None)
  smsgs = [m for m in msgs if m.which() in ['pandaStates', 'pandaStateDEPRECATED']]

  # Migrate safety param base on carState
  cp = [m for m in msgs if m.which() == 'carParams'][0].carParams
  if len(cp.safetyConfigs):
    safety_param = cp.safetyConfigs[0].safetyParam
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


def replay_sensor_events(s, msgs):
  pm = messaging.PubMaster([s, ])
  rk = Ratekeeper(service_list[s].frequency, print_delay_threshold=None)
  smsgs = [m for m in msgs if m.which() == s]
  while True:
    for m in smsgs:
      new_m = m.as_builder()
      new_m.logMonoTime = int(sec_since_boot() * 1e9)

      for evt in new_m.sensorEvents:
        evt.timestamp = new_m.logMonoTime

      pm.send(s, new_m)
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


def replay_cameras(lr, frs):
  eon_cameras = [
    ("roadCameraState", DT_MDL, eon_f_frame_size, VisionStreamType.VISION_STREAM_ROAD),
    ("driverCameraState", DT_DMON, eon_d_frame_size, VisionStreamType.VISION_STREAM_DRIVER),
  ]
  tici_cameras = [
    ("roadCameraState", DT_MDL, tici_f_frame_size, VisionStreamType.VISION_STREAM_ROAD),
    ("driverCameraState", DT_MDL, tici_d_frame_size, VisionStreamType.VISION_STREAM_DRIVER),
  ]

  def replay_camera(s, stream, dt, vipc_server, frames, size):
    pm = messaging.PubMaster([s, ])
    rk = Ratekeeper(1 / dt, print_delay_threshold=None)

    img = b"\x00" * int(size[0]*size[1]*3/2)
    while True:
      if frames is not None:
        img = frames[rk.frame % len(frames)]

      rk.keep_time()

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
  for (s, dt, size, stream) in cameras:
    fr = frs.get(s, None)

    frames = None
    if fr is not None:
      print(f"Decompressing frames {s}")
      frames = []
      for i in tqdm(range(fr.frame_count)):
        img = fr.get(i, pix_fmt='yuv420p')[0]
        frames.append(img.flatten().tobytes())

    vs.create_buffers(stream, 40, False, size[0], size[1])
    p.append(multiprocessing.Process(target=replay_camera,
                                     args=(s, stream, dt, vs, frames, size)))

  # hack to make UI work
  vs.create_buffers(VisionStreamType.VISION_STREAM_RGB_ROAD, 4, True, eon_f_frame_size[0], eon_f_frame_size[1])
  vs.start_listener()
  return vs, p


def regen_segment(lr, frs=None, outdir=FAKEDATA):
  lr = list(lr)
  if frs is None:
    frs = dict()

  setup_env()
  params = Params()

  os.environ["LOG_ROOT"] = outdir
  os.environ['SKIP_FW_QUERY'] = ""
  os.environ['FINGERPRINT'] = ""

  # TODO: remove after getting new route for mazda
  migration = {
    "Mazda CX-9 2021": "MAZDA CX-9 2021",
  }

  for msg in lr:
    if msg.which() == 'carParams':
      car_fingerprint = migration.get(msg.carParams.carFingerprint, msg.carParams.carFingerprint)
      if len(msg.carParams.carFw) and (car_fingerprint in FW_VERSIONS):
        params.put("CarParamsCache", msg.carParams.as_builder().to_bytes())
      else:
        os.environ['SKIP_FW_QUERY'] = "1"
        os.environ['FINGERPRINT'] = car_fingerprint
    elif msg.which() == 'liveCalibration':
      params.put("CalibrationParams", msg.as_builder().to_bytes())

  vs, cam_procs = replay_cameras(lr, frs)

  fake_daemons = {
    'sensord': [
      multiprocessing.Process(target=replay_sensor_events, args=('sensorEvents', lr)),
    ],
    'pandad': [
      multiprocessing.Process(target=replay_service, args=('can', lr)),
      multiprocessing.Process(target=replay_service, args=('ubloxRaw', lr)),
      multiprocessing.Process(target=replay_panda_states, args=('pandaStates', lr)),
    ],
    'managerState': [
     multiprocessing.Process(target=replay_manager_state, args=('managerState', lr)),
    ],
    'thermald': [
      multiprocessing.Process(target=replay_device_state, args=('deviceState', lr)),
    ],
    'camerad': [
      *cam_procs,
    ],
  }

  try:
    # start procs up
    ignore = list(fake_daemons.keys()) + ['ui', 'manage_athenad', 'uploader']
    ensure_running(managed_processes.values(), started=True, not_run=ignore)
    for procs in fake_daemons.values():
      for p in procs:
        p.start()

    for _ in tqdm(range(60)):
      # ensure all procs are running
      for d, procs in fake_daemons.items():
        for p in procs:
          if not p.is_alive():
            raise Exception(f"{d}'s {p.name} died")
      time.sleep(1)
  finally:
    # kill everything
    for p in managed_processes.values():
      p.stop()
    for procs in fake_daemons.values():
      for p in procs:
        p.terminate()

  del vs

  segment = params.get("CurrentRoute", encoding='utf-8') + "--0"
  seg_path = os.path.join(outdir, segment)
  # check to make sure openpilot is engaged in the route
  if not check_enabled(LogReader(os.path.join(seg_path, "rlog.bz2"))):
    raise Exception(f"Route never enabled: {segment}")

  return seg_path


def regen_and_save(route, sidx, upload=False, use_route_meta=False):
  if use_route_meta:
    r = Route(args.route)
    lr = LogReader(r.log_paths()[args.seg])
    fr = FrameReader(r.camera_paths()[args.seg])
  else:
    lr = LogReader(f"cd:/{route.replace('|', '/')}/{sidx}/rlog.bz2")
    fr = FrameReader(f"cd:/{route.replace('|', '/')}/{sidx}/fcamera.hevc")
  rpath = regen_segment(lr, {'roadCameraState': fr})

  lr = LogReader(os.path.join(rpath, 'rlog.bz2'))
  controls_state_active = [m.controlsState.active for m in lr if m.which() == 'controlsState']
  assert any(controls_state_active), "Segment did not engage"

  relr = os.path.relpath(rpath)

  print("\n\n", "*"*30, "\n\n")
  print("New route:", relr, "\n")
  if upload:
    upload_route(relr)
  return relr


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Generate new segments from old ones")
  parser.add_argument("--upload", action="store_true", help="Upload the new segment to the CI bucket")
  parser.add_argument("route", type=str, help="The source route")
  parser.add_argument("seg", type=int, help="Segment in source route")
  args = parser.parse_args()
  regen_and_save(args.route, args.seg, args.upload)
