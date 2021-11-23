#!/usr/bin/env python3
import os
import sys
import time
from typing import Any

from tqdm import tqdm

import cereal.messaging as messaging
from cereal import log
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
from selfdrive.version import get_git_commit
from tools.lib.framereader import FrameReader
from tools.lib.logreader import LogReader

if TICI:
  TEST_ROUTE = "4cf7a6ad03080c90|2021-09-29--13-46-36"
else:
  TEST_ROUTE = "303055c0002aefd1|2021-11-22--18-36-32"

CACHE_DIR = os.getenv("CACHE_DIR", None)

_VIPC_BUFS = [
  (VisionStreamType.VISION_STREAM_YUV_BACK, 40, False, *(tici_f_frame_size if TICI else eon_f_frame_size)),
  (VisionStreamType.VISION_STREAM_YUV_FRONT, 40, False, *(tici_d_frame_size if TICI else eon_d_frame_size))
]


def get_log_fn(ref_commit):
  return "%s_%s_%s.bz2" % (TEST_ROUTE, "model_tici" if TICI else "model", ref_commit)

def replace_calib(msg, calib):
  msg = msg.as_builder()
  if calib is not None:
    msg.liveCalibration.extrinsicMatrix = get_view_frame_from_road_frame(*calib, 1.22).flatten().tolist()
  return msg

def update_spinner(s, fidx, fcnt, didx, dcnt):
  s.update("replaying models:                    road %d/%d                    driver %d/%d" % (fidx, fcnt, didx, dcnt))

def model_replay(lr, fr, dfr, desire=None, calib=None):
  spinner = Spinner()
  spinner.update("starting model replay")

  vipc_server = VisionIpcServer("camerad")
  for vipc_buf in _VIPC_BUFS:
    vipc_server.create_buffers(*vipc_buf)
  vipc_server.start_listener()

  pm = messaging.PubMaster(['roadCameraState', 'driverCameraState', 'liveCalibration', 'lateralPlan'])
  sm = messaging.SubMaster(['modelV2', 'driverState'])

  try:
    managed_processes['modeld'].start()
    managed_processes['dmonitoringmodeld'].start()
    time.sleep(5)
    sm.update(1000)

    desires_by_index = {v:k for k,v in log.LateralPlan.Desire.schema.enumerants.items()}

    cal = [msg for msg in lr if msg.which() == "liveCalibration"]
    for msg in cal[:5]:
      pm.send(msg.which(), replace_calib(msg, calib))

    log_msgs = []
    frame_idx = 0
    dframe_idx = 0

    for msg in tqdm(lr):
      if msg.which() == "liveCalibration":
        pm.send(msg.which(), replace_calib(msg, calib))
      elif msg.which() == "roadCameraState":
        if desire is not None:
          for i in desire[frame_idx].nonzero()[0]:
            dat = messaging.new_message('lateralPlan')
            dat.lateralPlan.desire = desires_by_index[i]
            pm.send('lateralPlan', dat)

        f = msg.as_builder()
        pm.send(msg.which(), f)

        img = fr.get(frame_idx, pix_fmt="yuv420p")[0]
        vipc_server.send(VisionStreamType.VISION_STREAM_YUV_BACK, img.flatten().tobytes(), f.roadCameraState.frameId,
                         f.roadCameraState.timestampSof, f.roadCameraState.timestampEof)
        with Timeout(seconds=15):
          log_msgs.append(messaging.recv_one(sm.sock['modelV2']))

        frame_idx += 1
        if frame_idx >= fr.frame_count:
          break
        update_spinner(spinner, frame_idx, fr.frame_count, dframe_idx, dfr.frame_count)

      elif msg.which() == "driverCameraState":
        f = msg.as_builder()
        dimg = dfr.get(dframe_idx, pix_fmt="yuv420p")[0]
        vipc_server.send(VisionStreamType.VISION_STREAM_YUV_FRONT, dimg.flatten().tobytes(), f.driverCameraState.frameId,
                         f.driverCameraState.timestampSof, f.driverCameraState.timestampEof)
        with Timeout(seconds=15):
          log_msgs.append(messaging.recv_one(sm.sock['driverState']))

        dframe_idx += 1
        if dframe_idx >= dfr.frame_count:
          break
        update_spinner(spinner, frame_idx, fr.frame_count, dframe_idx, dfr.frame_count)

  except KeyboardInterrupt:
    pass
  finally:
    spinner.close()
    managed_processes['modeld'].stop()
    managed_processes['dmonitoringmodeld'].stop()

  return log_msgs

if __name__ == "__main__":

  update = "--update" in sys.argv

  replay_dir = os.path.dirname(os.path.abspath(__file__))
  ref_commit_fn = os.path.join(replay_dir, "model_replay_ref_commit")

  segnum = 0
  if CACHE_DIR:
    lr = LogReader(os.path.join(CACHE_DIR, '%s--%d--rlog.bz2' % (TEST_ROUTE.replace('|', '_'), segnum)))
    fr = FrameReader(os.path.join(CACHE_DIR, '%s--%d--fcamera.hevc' % (TEST_ROUTE.replace('|', '_'), segnum)))
    dfr = FrameReader(os.path.join(CACHE_DIR, '%s--%d--dcamera.hevc' % (TEST_ROUTE.replace('|', '_'), segnum)))
  else:
    lr = LogReader(get_url(TEST_ROUTE, segnum))
    fr = FrameReader(get_url(TEST_ROUTE, segnum, log_type="fcamera"))
    dfr = FrameReader(get_url(TEST_ROUTE, segnum, log_type="dcamera"))

  log_msgs = model_replay(list(lr), fr, dfr)

  failed = False
  if not update:
    ref_commit = open(ref_commit_fn).read().strip()
    log_fn = get_log_fn(ref_commit)
    cmp_log = LogReader(BASE_URL + log_fn)

    ignore = ['logMonoTime', 'valid',
              'modelV2.frameDropPerc',
              'modelV2.modelExecutionTime',
              'driverState.modelExecutionTime',
              'driverState.dspExecutionTime']
    tolerance = None if not PC else 1e-3
    results: Any = {TEST_ROUTE: {}}
    results[TEST_ROUTE]["models"] = compare_logs(cmp_log, log_msgs, tolerance=tolerance, ignore_fields=ignore)
    diff1, diff2, failed = format_diff(results, ref_commit)

    print(diff2)
    print('-------------')
    print('-------------')
    print('-------------')
    print('-------------')
    print('-------------')
    print(diff1)
    with open("model_diff.txt", "w") as f:
      f.write(diff2)

  if update or failed:
    from selfdrive.test.openpilotci import upload_file

    print("Uploading new refs")

    new_commit = get_git_commit()
    log_fn = get_log_fn(new_commit)
    save_log(log_fn, log_msgs)
    try:
      upload_file(log_fn, os.path.basename(log_fn))
    except Exception as e:
      print("failed to upload", e)

    with open(ref_commit_fn, 'w') as f:
      f.write(str(new_commit))

    print("\n\nNew ref commit: ", new_commit)

  sys.exit(int(failed))
