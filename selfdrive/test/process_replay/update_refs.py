#!/usr/bin/env python3
import os
import sys

from selfdrive.test.openpilotci import upload_file, get_url
from selfdrive.test.process_replay.compare_logs import save_log
from selfdrive.test.process_replay.process_replay import CONFIGS, PROC_REPLAY_DIR, CI, replay_process
from selfdrive.test.process_replay.test_processes import segments
from selfdrive.version import get_commit
from tools.lib.logreader import LogReader


if __name__ == "__main__":
  no_upload = "--no-upload" in sys.argv
  ref_commit_fn = os.path.join(PROC_REPLAY_DIR, "ref_commit")

  ref_commit = get_commit()
  if ref_commit is None:
    raise Exception("Couldn't get ref commit")
  with open(ref_commit_fn, "w") as f:
    f.write(ref_commit)

  # only upload
  if CI:
    for car_brand, segment in segments:
      for cfg in CONFIGS:
        log_fn = os.path.join(PROC_REPLAY_DIR, f"{segment}_{cfg.proc_name}_{ref_commit}.bz2")
        if not os.path.exists(log_fn):
          raise Exception(f"Cannot find log to upload: {log_fn}")

        print(f'Uploading: {log_fn}')
        upload_file(log_fn, os.path.basename(log_fn))
        os.remove(log_fn)
  else:
    for car_brand, segment in segments:
      r, n = segment.rsplit("--", 1)
      lr = LogReader(get_url(r, n))

      for cfg in CONFIGS:
        log_msgs = replay_process(cfg, lr)
        log_fn = os.path.join(PROC_REPLAY_DIR, f"{segment}_{cfg.proc_name}_{ref_commit}.bz2")
        save_log(log_fn, log_msgs)

        if not no_upload:
          upload_file(log_fn, os.path.basename(log_fn))
          os.remove(log_fn)

  print(f'Done\nNew reference commit: {ref_commit}')
