#!/usr/bin/env python3
import os
import sys
import time
from typing import Any
from tqdm import tqdm

from common.hardware import ANDROID
os.environ['CI'] = "1"
if ANDROID:
  os.environ['QCOM_REPLAY'] = "1"

from common.timeout import Timeout
import selfdrive.manager as manager

from common.spinner import Spinner
import cereal.messaging as messaging
from tools.lib.framereader import FrameReader
from tools.lib.logreader import LogReader
from selfdrive.test.openpilotci import BASE_URL, get_url
from selfdrive.test.process_replay.compare_logs import compare_logs, save_log
from selfdrive.test.process_replay.test_processes import format_diff
from selfdrive.version import get_git_commit

TEST_ROUTE = "99c94dc769b5d96e|2019-08-03--14-19-59"

def camera_replay(lr, fr):

  spinner = Spinner()

  pm = messaging.PubMaster(['frame', 'liveCalibration'])
  sm = messaging.SubMaster(['model'])

  # TODO: add dmonitoringmodeld
  print("preparing procs")
  manager.prepare_managed_process("camerad")
  manager.prepare_managed_process("modeld")
  try:
    print("starting procs")
    manager.start_managed_process("camerad")
    manager.start_managed_process("modeld")
    time.sleep(5)
    print("procs started")

    cal = [msg for msg in lr if msg.which() == "liveCalibration"]
    for msg in cal[:5]:
      pm.send(msg.which(), msg.as_builder())

    log_msgs = []
    frame_idx = 0
    for msg in tqdm(lr):
      if msg.which() == "liveCalibrationd":
        pm.send(msg.which(), msg.as_builder())
      elif msg.which() == "frame":
        f = msg.as_builder()
        img = fr.get(frame_idx, pix_fmt="rgb24")[0][:, ::, -1]
        f.frame.image = img.flatten().tobytes()
        frame_idx += 1

        pm.send(msg.which(), f)
        with Timeout(seconds=15):
          log_msgs.append(messaging.recv_one(sm.sock['model']))

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

  lr = LogReader(get_url(TEST_ROUTE, 0))
  fr = FrameReader(get_url(TEST_ROUTE, 0, log_type="fcamera"))

  log_msgs = camera_replay(list(lr), fr)

  failed = False
  if not update:
    ref_commit = open("model_replay_ref_commit").read().strip()
    log_fn = "%s_%s_%s.bz2" % (TEST_ROUTE, "model", ref_commit)
    cmp_log = LogReader(BASE_URL + log_fn)
    results: Any = {TEST_ROUTE: {}}
    results[TEST_ROUTE]["modeld"] = compare_logs(cmp_log, log_msgs, ignore_fields=['logMonoTime', 'valid', 'model.frameDropPerc', 'model.modelExecutionTime'])
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
    upload_file(log_fn, os.path.basename(log_fn))

    print("\n\nNew ref commit: ", new_commit)

  sys.exit(int(failed))
