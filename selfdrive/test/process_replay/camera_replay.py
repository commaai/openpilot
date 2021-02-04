#!/usr/bin/env python3
import os
import sys
import time
from typing import Any
from tqdm import tqdm

os.environ['QCOM_REPLAY'] = '1'

from common.spinner import Spinner
from common.timeout import Timeout
import selfdrive.manager as manager

from cereal import log
import cereal.messaging as messaging
from tools.lib.framereader import FrameReader
from tools.lib.logreader import LogReader
from selfdrive.test.openpilotci import BASE_URL, get_url
from selfdrive.test.process_replay.compare_logs import compare_logs, save_log
from selfdrive.test.process_replay.test_processes import format_diff
from selfdrive.version import get_git_commit
from common.transformations.camera import get_view_frame_from_road_frame

TEST_ROUTE = "99c94dc769b5d96e|2019-08-03--14-19-59"

def replace_calib(msg, calib):
  msg = msg.as_builder()
  if calib is not None:
    msg.liveCalibration.extrinsicMatrix = get_view_frame_from_road_frame(*calib, 1.22).flatten().tolist()
  return msg

def camera_replay(lr, fr, desire=None, calib=None):

  spinner = Spinner()
  spinner.update("starting model replay")

  pm = messaging.PubMaster(['frame', 'liveCalibration', 'lateralPlan'])
  sm = messaging.SubMaster(['modelV2'])

  # TODO: add dmonitoringmodeld
  print("preparing procs")
  manager.prepare_managed_process("camerad")
  manager.prepare_managed_process("modeld")
  try:
    print("starting procs")
    manager.start_managed_process("camerad")
    manager.start_managed_process("modeld")
    time.sleep(5)
    sm.update(1000)
    print("procs started")

    desires_by_index = {v:k for k,v in log.LateralPlan.Desire.schema.enumerants.items()}

    cal = [msg for msg in lr if msg.which() == "liveCalibration"]
    for msg in cal[:5]:
      pm.send(msg.which(), replace_calib(msg, calib))

    log_msgs = []
    frame_idx = 0
    for msg in tqdm(lr):
      if msg.which() == "liveCalibration":
        pm.send(msg.which(), replace_calib(msg, calib))
      elif msg.which() == "frame":
        if desire is not None:
          for i in desire[frame_idx].nonzero()[0]:
            dat = messaging.new_message('lateralPlan')
            dat.lateralPlan.desire = desires_by_index[i]
            pm.send('lateralPlan', dat)

        f = msg.as_builder()
        img = fr.get(frame_idx, pix_fmt="rgb24")[0][:,:,::-1]
        f.frame.image = img.flatten().tobytes()
        frame_idx += 1

        pm.send(msg.which(), f)
        with Timeout(seconds=15):
          log_msgs.append(messaging.recv_one(sm.sock['modelV2']))

        spinner.update("modeld replay %d/%d" % (frame_idx, fr.frame_count))

        if frame_idx >= fr.frame_count:
          break
  except KeyboardInterrupt:
    pass

  print("replay done")
  spinner.close()
  manager.kill_managed_process('modeld')
  time.sleep(2)
  manager.kill_managed_process('camerad')
  return log_msgs

if __name__ == "__main__":

  update = "--update" in sys.argv

  replay_dir = os.path.dirname(os.path.abspath(__file__))
  ref_commit_fn = os.path.join(replay_dir, "model_replay_ref_commit")

  lr = LogReader(get_url(TEST_ROUTE, 0))
  fr = FrameReader(get_url(TEST_ROUTE, 0, log_type="fcamera"))

  log_msgs = camera_replay(list(lr), fr)

  failed = False
  if not update:
    ref_commit = open(ref_commit_fn).read().strip()
    log_fn = "%s_%s_%s.bz2" % (TEST_ROUTE, "model", ref_commit)
    cmp_log = LogReader(BASE_URL + log_fn)

    ignore = ['logMonoTime', 'valid',
              'modelV2.frameDropPerc',
              'modelV2.modelExecutionTime']
    results: Any = {TEST_ROUTE: {}}
    results[TEST_ROUTE]["modeld"] = compare_logs(cmp_log, log_msgs, ignore_fields=ignore)
    diff1, diff2, failed = format_diff(results, ref_commit)

    print(diff1)
    with open("model_diff.txt", "w") as f:
      f.write(diff2)

  if update or failed:
    from selfdrive.test.openpilotci import upload_file

    print("Uploading new refs")

    new_commit = get_git_commit()
    log_fn = "%s_%s_%s.bz2" % (TEST_ROUTE, "model", new_commit)
    save_log(log_fn, log_msgs)
    try:
      upload_file(log_fn, os.path.basename(log_fn))
    except Exception as e:
      print("failed to upload", e)

    with open(ref_commit_fn, 'w') as f:
      f.write(str(new_commit))

    print("\n\nNew ref commit: ", new_commit)

  sys.exit(int(failed))
