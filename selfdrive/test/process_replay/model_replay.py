#!/usr/bin/env python3
import os
import sys
import time
from collections import defaultdict
from typing import Any
from itertools import zip_longest

import cereal.messaging as messaging
from cereal.visionipc.visionipc_pyx import VisionIpcServer, VisionStreamType  # pylint: disable=no-name-in-module, import-error
from common.spinner import Spinner
from common.timeout import Timeout
from common.transformations.camera import get_view_frame_from_road_frame, eon_f_frame_size, tici_f_frame_size, \
                                          eon_d_frame_size, tici_d_frame_size
from selfdrive.hardware import PC, TICI
from selfdrive.manager.process_config import managed_processes
from selfdrive.test.openpilotci import BASE_URL, get_url
from selfdrive.test.process_replay.compare_logs import compare_logs, save_log
from selfdrive.test.process_replay.test_processes import format_diff
from selfdrive.version import get_commit
from tools.lib.framereader import FrameReader
from tools.lib.logreader import LogReader

TICI_TEST_ROUTE = "4cf7a6ad03080c90|2021-09-29--13-46-36"
EON_TEST_ROUTE = "303055c0002aefd1|2021-11-22--18-36-32"
SEGMENT = 0
if TICI:
  TEST_ROUTE = TICI_TEST_ROUTE
else:
  TEST_ROUTE = EON_TEST_ROUTE

SEND_EXTRA_INPUTS = bool(os.getenv("SEND_EXTRA_INPUTS", "0"))

VIPC_STREAM = {"roadCameraState": VisionStreamType.VISION_STREAM_ROAD, "driverCameraState": VisionStreamType.VISION_STREAM_DRIVER,
               "wideRoadCameraState": VisionStreamType.VISION_STREAM_WIDE_ROAD}

def get_log_fn(ref_commit, test_route, tici=True):
  return f"{test_route}_{'model_tici' if tici else 'model'}_{ref_commit}.bz2"


def replace_calib(msg, calib):
  msg = msg.as_builder()
  if calib is not None:
    msg.liveCalibration.extrinsicMatrix = get_view_frame_from_road_frame(*calib, 1.22).flatten().tolist()
  return msg


def model_replay(lr, frs):
  spinner = Spinner()
  spinner.update("starting model replay")

  vipc_server = VisionIpcServer("camerad")
  vipc_server.create_buffers(VisionStreamType.VISION_STREAM_ROAD, 40, False, *(tici_f_frame_size if TICI else eon_f_frame_size))
  vipc_server.create_buffers(VisionStreamType.VISION_STREAM_DRIVER, 40, False, *(tici_d_frame_size if TICI else eon_d_frame_size))
  vipc_server.create_buffers(VisionStreamType.VISION_STREAM_WIDE_ROAD, 40, False, *(tici_f_frame_size))
  vipc_server.start_listener()

  sm = messaging.SubMaster(['modelV2', 'driverState'])
  pm = messaging.PubMaster(['roadCameraState', 'wideRoadCameraState', 'driverCameraState', 'liveCalibration', 'lateralPlan'])

  try:
    managed_processes['modeld'].start()
    managed_processes['dmonitoringmodeld'].start()
    time.sleep(5)
    sm.update(1000)

    log_msgs = []
    last_desire = None
    recv_cnt = defaultdict(lambda: 0)
    frame_idxs = defaultdict(lambda: 0)

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
      if TICI and None in (cam_msgs[0], cam_msgs[1]):
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
          img = frs[msg.which()].get(frame_idxs[msg.which()], pix_fmt="yuv420p")[0]
          frame_idxs[msg.which()] += 1

          # send camera state and frame
          camera_state.frameId = frame_idxs[msg.which()]
          pm.send(msg.which(), msg)
          vipc_server.send(VIPC_STREAM[msg.which()], img.flatten().tobytes(), camera_state.frameId,
                           camera_state.timestampSof, camera_state.timestampEof)

          recv = None
          if msg.which() in ('roadCameraState', 'wideRoadCameraState'):
            if not TICI or min(frame_idxs['roadCameraState'], frame_idxs['wideRoadCameraState']) > recv_cnt['modelV2']:
              recv = "modelV2"
          elif msg.which() == 'driverCameraState':
            recv = "driverState"

          # wait for a response
          with Timeout(15, f"timed out waiting for {recv}"):
            if recv:
              recv_cnt[recv] += 1
              log_msgs.append(messaging.recv_one(sm.sock[recv]))

          spinner.update("replaying models:  road %d/%d,  driver %d/%d" % (frame_idxs['roadCameraState'],
                         frs['roadCameraState'].frame_count, frame_idxs['driverCameraState'], frs['driverCameraState'].frame_count))

      if any(frame_idxs[c] >= frs[c].frame_count for c in frame_idxs.keys()):
        break

  finally:
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
    'roadCameraState': FrameReader(get_url(TEST_ROUTE, SEGMENT, log_type="fcamera")),
    'driverCameraState': FrameReader(get_url(TEST_ROUTE, SEGMENT, log_type="dcamera")),
  }
  if TICI:
    frs['wideRoadCameraState'] = FrameReader(get_url(TEST_ROUTE, SEGMENT, log_type="ecamera"))

  # run replay
  log_msgs = model_replay(lr, frs)

  # get diff
  failed = False
  if not update:
    with open(ref_commit_fn) as f:
      ref_commit = f.read().strip()
    log_fn = get_log_fn(ref_commit, TEST_ROUTE, tici=TICI)
    try:
      cmp_log = LogReader(BASE_URL + log_fn)

      ignore = [
        'logMonoTime',
        'modelV2.frameDropPerc',
        'modelV2.modelExecutionTime',
        'driverState.modelExecutionTime',
        'driverState.dspExecutionTime'
      ]
      tolerance = None if not PC else 1e-3
      results: Any = {TEST_ROUTE: {}}
      results[TEST_ROUTE]["models"] = compare_logs(cmp_log, log_msgs, tolerance=tolerance, ignore_fields=ignore)
      diff1, diff2, failed = format_diff(results, ref_commit)

      print(diff2)
      print('-------------\n'*5)
      print(diff1)
      with open("model_diff.txt", "w") as f:
        f.write(diff2)
    except Exception as e:
      print(str(e))
      failed = True

  # upload new refs
  if update or failed:
    from selfdrive.test.openpilotci import upload_file

    print("Uploading new refs")

    new_commit = get_commit()
    log_fn = get_log_fn(new_commit, TEST_ROUTE, tici=TICI)
    save_log(log_fn, log_msgs)
    try:
      upload_file(log_fn, os.path.basename(log_fn))
    except Exception as e:
      print("failed to upload", e)

    with open(ref_commit_fn, 'w') as f:
      f.write(str(new_commit))

    print("\n\nNew ref commit: ", new_commit)

  sys.exit(int(failed))
