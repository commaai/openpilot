#!/usr/bin/env python3
import argparse
import os

from selfdrive.test.openpilotci import upload_file, get_url
from selfdrive.test.process_replay.compare_logs import save_log
from selfdrive.test.process_replay.process_replay import replay_process, CONFIGS
from selfdrive.test.process_replay.test_processes import segments
from selfdrive.version import get_commit
from tools.lib.logreader import LogReader

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Updates the reference logs for the current commit")

  parser.add_argument("--no-upload", action="store_true")
  parser.add_argument("--only-upload", action="store_true")  # TODO: split this out into own file upload_refs?
  args = parser.parse_args()
  assert args.no_upload != args.only_upload or not args.no_upload, "Both upload args can't be set"

  process_replay_dir = os.path.dirname(os.path.abspath(__file__))
  ref_commit_fn = os.path.join(process_replay_dir, "ref_commit")

  ref_commit = get_commit()
  if ref_commit is None:
    raise Exception("couldn't get ref commit")
  with open(ref_commit_fn, "w") as f:
    f.write(ref_commit)

  for car_brand, segment in segments:
    if args.only_upload:
      for cfg in CONFIGS:
        log_fn = os.path.join(process_replay_dir, f"{segment}_{cfg.proc_name}_{ref_commit}.bz2")
        if not os.path.exists(log_fn):
          raise Exception("couldn't find file for uploading: {}".format(log_fn))
        upload_file(log_fn, os.path.basename(log_fn))
        os.remove(log_fn)
      continue

    r, n = segment.rsplit("--", 1)
    lr = LogReader(get_url(r, n))

    for cfg in CONFIGS:
      log_msgs = replay_process(cfg, lr)
      log_fn = os.path.join(process_replay_dir, f"{segment}_{cfg.proc_name}_{ref_commit}.bz2")
      save_log(log_fn, log_msgs)

      if not args.no_upload:
        upload_file(log_fn, os.path.basename(log_fn))
        os.remove(log_fn)

  print("done")
