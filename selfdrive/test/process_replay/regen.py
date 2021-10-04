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
from common.realtime import Ratekeeper, DT_MDL, DT_DMON
from common.transformations.camera import eon_f_frame_size, eon_d_frame_size
from selfdrive.car.fingerprints import FW_VERSIONS
from selfdrive.manager.process import ensure_running
from selfdrive.manager.process_config import managed_processes
from selfdrive.test.update_ci_routes import upload_route
from tools.lib.route import Route
from tools.lib.framereader import FrameReader
from tools.lib.logreader import LogReader


process_replay_dir = os.path.dirname(os.path.abspath(__file__))
FAKEDATA = os.path.join(process_replay_dir, "fakedata/")


def replay_service(s, msgs):
  pm = messaging.PubMaster([s, ])
  rk = Ratekeeper(service_list[s].frequency, print_delay_threshold=None)
  smsgs = [m for m in msgs if m.which() == s]
  while True:
    for m in smsgs:
      # TODO: use logMonoTime
      pm.send(s, m.as_builder())
      rk.keep_time()

vs = None
def replay_cameras(lr, frs):
  cameras = [
    ("roadCameraState", DT_MDL, eon_f_frame_size, VisionStreamType.VISION_STREAM_YUV_BACK),
    ("driverCameraState", DT_DMON, eon_d_frame_size, VisionStreamType.VISION_STREAM_YUV_FRONT),
  ]

  def replay_camera(s, stream, dt, vipc_server, fr, size):
    pm = messaging.PubMaster([s, ])
    rk = Ratekeeper(1 / dt, print_delay_threshold=None)

    img = b"\x00" * int(size[0]*size[1]*3/2)
    while True:
      if fr is not None:
        img = fr.get(rk.frame % fr.frame_count, pix_fmt='yuv420p')[0]
        img = img.flatten().tobytes()

      rk.keep_time()

      m = messaging.new_message(s)
      msg = getattr(m, s)
      msg.frameId = rk.frame
      pm.send(s, m)

      vipc_server.send(stream, img, msg.frameId, msg.timestampSof, msg.timestampEof)

  # init vipc server and cameras
  p = []
  global vs
  vs = VisionIpcServer("camerad")
  for (s, dt, size, stream) in cameras:
    fr = frs.get(s, None)
    vs.create_buffers(stream, 40, False, size[0], size[1])
    p.append(multiprocessing.Process(target=replay_camera,
                                     args=(s, stream, dt, vs, fr, size)))

  # hack to make UI work
  vs.create_buffers(VisionStreamType.VISION_STREAM_RGB_BACK, 4, True, eon_f_frame_size[0], eon_f_frame_size[1])
  vs.start_listener()
  return p


def regen_segment(lr, frs=None, outdir=FAKEDATA):

  lr = list(lr)
  if frs is None:
    frs = dict()

  # setup env
  params = Params()
  params.clear_all()
  params.put_bool("Passive", False)
  params.put_bool("OpenpilotEnabledToggle", True)
  params.put_bool("CommunityFeaturesToggle", True)
  params.put_bool("CommunityFeaturesToggle", True)
  cal = messaging.new_message('liveCalibration')
  cal.liveCalibration.validBlocks = 20
  cal.liveCalibration.rpyCalib = [0.0, 0.0, 0.0]
  params.put("CalibrationParams", cal.to_bytes())

  os.environ["LOG_ROOT"] = outdir
  os.environ["SIMULATION"] = "1"

  os.environ['SKIP_FW_QUERY'] = ""
  os.environ['FINGERPRINT'] = ""
  for msg in lr:
    if msg.which() == 'carParams':
      car_fingerprint = msg.carParams.carFingerprint
      if len(msg.carParams.carFw) and (car_fingerprint in FW_VERSIONS):
        params.put("CarParamsCache", msg.carParams.as_builder().to_bytes())
      else:
        os.environ['SKIP_FW_QUERY'] = "1"
        os.environ['FINGERPRINT'] = car_fingerprint

  #TODO: init car, make sure starts engaged when segment is engaged

  fake_daemons = {
    'sensord': [
      multiprocessing.Process(target=replay_service, args=('sensorEvents', lr)),
    ],
    'pandad': [
      multiprocessing.Process(target=replay_service, args=('can', lr)),
      multiprocessing.Process(target=replay_service, args=('pandaState', lr)),
    ],
    #'managerState': [
    #  multiprocessing.Process(target=replay_service, args=('managerState', lr)),
    #],
    'thermald': [
      multiprocessing.Process(target=replay_service, args=('deviceState', lr)),
    ],
    'camerad': [
      *replay_cameras(lr, frs),
    ],

    # TODO: fix these and run them
    'paramsd': [
      multiprocessing.Process(target=replay_service, args=('liveParameters', lr)),
    ],
    'locationd': [
      multiprocessing.Process(target=replay_service, args=('liveLocationKalman', lr)),
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

  r = params.get("CurrentRoute", encoding='utf-8')
  return os.path.join(outdir, f"{r}--0")


def regen_and_save(route, sidx, upload=False, use_route_meta=True):
  if use_route_meta:
    r = Route(args.route)
    lr = LogReader(r.log_paths()[args.seg])
    fr = FrameReader(r.camera_paths()[args.seg])
  else:
    lr = LogReader(f"cd:/{route.replace('|', '/')}/{sidx}/rlog.bz2")
    fr = FrameReader(f"cd:/{route.replace('|', '/')}/{sidx}/fcamera.hevc")
  rpath = regen_segment(lr, {'roadCameraState': fr})
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
