#!/usr/bin/env python2
import os
import sys

from selfdrive.test.openpilotci_upload import upload_file
from selfdrive.test.tests.process_replay.compare_logs import save_log
from selfdrive.test.tests.process_replay.process_replay import replay_process, CONFIGS
from selfdrive.test.tests.process_replay.test_processes import segments, get_segment
from selfdrive.version import get_git_commit
from tools.lib.logreader import LogReader

if __name__ == "__main__":

  no_upload = "--no-upload" in sys.argv

  process_replay_dir = os.path.dirname(os.path.abspath(__file__))
  ref_commit_fn = os.path.join(process_replay_dir, "ref_commit")

  ref_commit = get_git_commit()
  with open(ref_commit_fn, "w") as f:
    f.write(ref_commit)

  for segment in segments:
    rlog_fn = get_segment(segment)

    if rlog_fn is None:
      print "failed to get segment %s" % segment
      sys.exit(1)

    lr = LogReader(rlog_fn)

    for cfg in CONFIGS:
      log_msgs = replay_process(cfg, lr)
      log_fn = os.path.join(process_replay_dir, "%s_%s_%s.bz2" % (segment, cfg.proc_name, ref_commit))
      save_log(log_fn, log_msgs)

      if not no_upload:
        upload_file(log_fn, os.path.basename(log_fn))
        os.remove(log_fn)
    os.remove(rlog_fn)

  print "done"
