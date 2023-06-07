#!/usr/bin/env python3
import os
import sys
import time
from collections import defaultdict
from typing import Any
from itertools import zip_longest

import cereal.messaging as messaging
from cereal.visionipc import VisionIpcServer, VisionStreamType
from common.spinner import Spinner
from common.timeout import Timeout
from common.transformations.camera import tici_f_frame_size, tici_d_frame_size
from system.hardware import PC
from selfdrive.manager.process_config import managed_processes
from selfdrive.test.openpilotci import BASE_URL, get_url
from selfdrive.test.process_replay.compare_logs import compare_logs
from selfdrive.test.process_replay.test_processes import format_diff
from system.version import get_commit
from tools.lib.framereader import FrameReader
from tools.lib.logreader import LogReader
from tools.lib.helpers import save_log

TEST_ROUTE = "2f4452b03ccb98f0|2022-12-03--13-45-30"
SEGMENT = 6
MAX_FRAMES = 100 if PC else 600
NAV_FRAMES = 50

NO_NAV = "NO_NAV" in os.environ
SEND_EXTRA_INPUTS = bool(os.getenv("SEND_EXTRA_INPUTS", "0"))

VIPC_STREAM = {"roadCameraState": VisionStreamType.VISION_STREAM_ROAD, "driverCameraState": VisionStreamType.VISION_STREAM_DRIVER,
               "wideRoadCameraState": VisionStreamType.VISION_STREAM_WIDE_ROAD}

def get_log_fn(ref_commit, test_route):
  return f"{test_route}_model_tici_{ref_commit}.bz2"


def replace_calib(msg, calib):
  msg = msg.as_builder()
  if calib is not None:
    msg.liveCalibration.rpyCalib = calib.tolist()
  return msg


def nav_model_replay(lr):
  sm = messaging.SubMaster(['navModel', 'navThumbnail', 'mapRenderState'])
  pm = messaging.PubMaster(['liveLocationKalman', 'navRoute'])

  nav = [m for m in lr if m.which() == 'navRoute']
  llk = [m for m in lr if m.which() == 'liveLocationKalman']
  assert len(nav) > 0 and len(llk) >= NAV_FRAMES and nav[0].logMonoTime < llk[-NAV_FRAMES].logMonoTime

  log_msgs = []
  try:
    assert "MAPBOX_TOKEN" in os.environ
    os.environ['MAP_RENDER_TEST_MODE'] = '1'
    managed_processes['mapsd'].start()
    managed_processes['navmodeld'].start()

    # setup position and route
    for _ in range(10):
      for s in (llk[-NAV_FRAMES], nav[0]):
        pm.send(s.which(), s.as_builder().to_bytes())
      sm.update(1000)
      if sm.updated['navModel']:
        break
      time.sleep(1)

    if not sm.updated['navModel']:
      raise Exception("no navmodeld outputs, failed to initialize")

    # drain
    time.sleep(2)
    sm.update(0)

    # run replay
    for n in range(len(llk) - NAV_FRAMES, len(llk)):
      pm.send(llk[n].which(), llk[n].as_builder().to_bytes())
      m = messaging.recv_one(sm.sock['navThumbnail'])
      assert m is not None, f"no navThumbnail, frame={n}"
      log_msgs.append(m)

      m = messaging.recv_one(sm.sock['mapRenderState'])
      assert m is not None, f"no mapRenderState, frame={n}"
      log_msgs.append(m)

      m = messaging.recv_one(sm.sock['navModel'])
      assert m is not None, f"no navModel response, frame={n}"
      log_msgs.append(m)
  finally:
    managed_processes['mapsd'].stop()
    managed_processes['navmodeld'].stop()

  return log_msgs


def model_replay(lr, frs):
  if not PC:
    spinner = Spinner()
    spinner.update("starting model replay")
  else:
    spinner = None

  vipc_server = VisionIpcServer("camerad")
  vipc_server.create_buffers(VisionStreamType.VISION_STREAM_ROAD, 40, False, *(tici_f_frame_size))
  vipc_server.create_buffers(VisionStreamType.VISION_STREAM_DRIVER, 40, False, *(tici_d_frame_size))
  vipc_server.create_buffers(VisionStreamType.VISION_STREAM_WIDE_ROAD, 40, False, *(tici_f_frame_size))
  vipc_server.start_listener()

  sm = messaging.SubMaster(['modelV2', 'driverStateV2'])
  pm = messaging.PubMaster(['roadCameraState', 'wideRoadCameraState', 'driverCameraState', 'liveCalibration', 'lateralPlan'])

  try:
    managed_processes['modeld'].start()
    managed_processes['dmonitoringmodeld'].start()
    time.sleep(5)
    sm.update(1000)

    log_msgs = []
    last_desire = None
    recv_cnt = defaultdict(int)
    frame_idxs = defaultdict(int)

    # init modeld with valid calibration
    cal_msgs = [msg for msg in lr if msg.which() == "liveCalibration"]
    for _ in range(5):
      pm.send(cal_msgs[0].which(), cal_msgs[0].as_builder())
      time.sleep(0.1)

    msgs = defaultdict(list)
    for msg in lr:
      msgs[msg.which()].append(msg)

    for cam_msgs in zip_longest(msgs['roadCameraState'], msgs['wideRoadCameraState'], msgs['driverCameraState']):
      # need a pair of road/wide msgs
      if None in (cam_msgs[0], cam_msgs[1]):
        break

      for msg in cam_msgs:
        if msg is None:
          continue

        if SEND_EXTRA_INPUTS:
          if msg.which() == "liveCalibration":
            last_calib = list(msg.liveCalibration.rpyCalib)
            pm.send(msg.which(), replace_calib(msg, last_calib))
          elif msg.which() == "lateralPlan":
            last_desire = msg.lateralPlan.desire
            dat = messaging.new_message('lateralPlan')
            dat.lateralPlan.desire = last_desire
            pm.send('lateralPlan', dat)

        if msg.which() in VIPC_STREAM:
          msg = msg.as_builder()
          camera_state = getattr(msg, msg.which())
          img = frs[msg.which()].get(frame_idxs[msg.which()], pix_fmt="nv12")[0]
          frame_idxs[msg.which()] += 1

          # send camera state and frame
          camera_state.frameId = frame_idxs[msg.which()]
          pm.send(msg.which(), msg)
          vipc_server.send(VIPC_STREAM[msg.which()], img.flatten().tobytes(), camera_state.frameId,
                           camera_state.timestampSof, camera_state.timestampEof)

          recv = None
          if msg.which() in ('roadCameraState', 'wideRoadCameraState'):
            if min(frame_idxs['roadCameraState'], frame_idxs['wideRoadCameraState']) > recv_cnt['modelV2']:
              recv = "modelV2"
          elif msg.which() == 'driverCameraState':
            recv = "driverStateV2"

          # wait for a response
          with Timeout(15, f"timed out waiting for {recv}"):
            if recv:
              recv_cnt[recv] += 1
              log_msgs.append(messaging.recv_one(sm.sock[recv]))

          if spinner:
            spinner.update("replaying models:  road %d/%d,  driver %d/%d" % (frame_idxs['roadCameraState'],
                           frs['roadCameraState'].frame_count, frame_idxs['driverCameraState'], frs['driverCameraState'].frame_count))


      if any(frame_idxs[c] >= frs[c].frame_count for c in frame_idxs.keys()) or frame_idxs['roadCameraState'] == MAX_FRAMES:
        break
      else:
        print(f'Received {frame_idxs["roadCameraState"]} frames')

  finally:
    if spinner:
      spinner.close()
    managed_processes['modeld'].stop()
    managed_processes['dmonitoringmodeld'].stop()


  return log_msgs


if __name__ == "__main__":

  update = "--update" in sys.argv
  replay_dir = os.path.dirname(os.path.abspath(__file__))
  ref_commit_fn = os.path.join(replay_dir, "model_replay_ref_commit")

  # load logs
  lr = list(LogReader(get_url(TEST_ROUTE, SEGMENT)))
  frs = {
    'roadCameraState': FrameReader(get_url(TEST_ROUTE, SEGMENT, log_type="fcamera"), readahead=True),
    'driverCameraState': FrameReader(get_url(TEST_ROUTE, SEGMENT, log_type="dcamera"), readahead=True),
    'wideRoadCameraState': FrameReader(get_url(TEST_ROUTE, SEGMENT, log_type="ecamera"), readahead=True)
  }

  # Update tile refs
  if update:
    import urllib
    import requests
    import threading
    import http.server
    from selfdrive.test.openpilotci import upload_bytes
    os.environ['MAPS_HOST'] = 'http://localhost:5000'

    class HTTPRequestHandler(http.server.BaseHTTPRequestHandler):
      def do_GET(self):
        assert len(self.path) > 10  # Sanity check on path length
        r = requests.get(f'https://api.mapbox.com{self.path}', timeout=30)
        upload_bytes(r.content, urllib.parse.urlparse(self.path).path.lstrip('/'))
        self.send_response(r.status_code)
        self.send_header('Content-type','text/html')
        self.end_headers()
        self.wfile.write(r.content)

    server = http.server.HTTPServer(("127.0.0.1", 5000), HTTPRequestHandler)
    thread = threading.Thread(None, server.serve_forever, daemon=True)
    thread.start()
  else:
    os.environ['MAPS_HOST'] = BASE_URL.rstrip('/')

  # run replays
  log_msgs = model_replay(lr, frs)
  if not NO_NAV:
    log_msgs += nav_model_replay(lr)

  # get diff
  failed = False
  if not update:
    with open(ref_commit_fn) as f:
      ref_commit = f.read().strip()
    log_fn = get_log_fn(ref_commit, TEST_ROUTE)
    try:
      expected_msgs = 2*MAX_FRAMES
      if not NO_NAV:
        expected_msgs += NAV_FRAMES*3
      cmp_log = list(LogReader(BASE_URL + log_fn))[:expected_msgs]

      ignore = [
        'logMonoTime',
        'modelV2.frameDropPerc',
        'modelV2.modelExecutionTime',
        'driverStateV2.modelExecutionTime',
        'driverStateV2.dspExecutionTime',
        'navModel.dspExecutionTime',
        'navModel.modelExecutionTime',
        'navThumbnail.timestampEof',
        'mapRenderState.locationMonoTime',
        'mapRenderState.renderTime',
      ]
      if PC:
        ignore += [
          'modelV2.laneLines.0.t',
          'modelV2.laneLines.1.t',
          'modelV2.laneLines.2.t',
          'modelV2.laneLines.3.t',
          'modelV2.roadEdges.0.t',
          'modelV2.roadEdges.1.t',
        ]
      # TODO this tolerance is absurdly large
      tolerance = 2.0 if PC else None
      results: Any = {TEST_ROUTE: {}}
      log_paths: Any = {TEST_ROUTE: {"models": {'ref': BASE_URL + log_fn, 'new': log_fn}}}
      results[TEST_ROUTE]["models"] = compare_logs(cmp_log, log_msgs, tolerance=tolerance, ignore_fields=ignore)
      diff1, diff2, failed = format_diff(results, log_paths, ref_commit)

      print(diff2)
      print('-------------\n'*5)
      print(diff1)
      with open("model_diff.txt", "w") as f:
        f.write(diff2)
    except Exception as e:
      print(str(e))
      failed = True

  # upload new refs
  if (update or failed) and not PC:
    from selfdrive.test.openpilotci import upload_file

    print("Uploading new refs")

    new_commit = get_commit()
    log_fn = get_log_fn(new_commit, TEST_ROUTE)
    save_log(log_fn, log_msgs)
    try:
      upload_file(log_fn, os.path.basename(log_fn))
    except Exception as e:
      print("failed to upload", e)

    with open(ref_commit_fn, 'w') as f:
      f.write(str(new_commit))

    print("\n\nNew ref commit: ", new_commit)

  sys.exit(int(failed))
