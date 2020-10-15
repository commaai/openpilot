#!/usr/bin/env python3
import os
import sys

from selfdrive.test.openpilotci import upload_file
from selfdrive.test.process_replay.compare_logs import save_log
from selfdrive.test.process_replay.test_processes import segments, get_segment
from selfdrive.version import get_git_commit
from tools.lib.logreader import LogReader
from inject_model import inject_model

if __name__ == "__main__":

  no_upload = "--no-upload" in sys.argv

  process_replay_dir = os.path.dirname(os.path.abspath(__file__))
  ref_commit_fn = os.path.join(process_replay_dir, "model_ref_commit")

  ref_commit = get_git_commit()
  if ref_commit is None:
    raise Exception("couldn't get ref commit")
  with open(ref_commit_fn, "w") as f:
    f.write(ref_commit)

  for car_brand, segment in segments:
    rlog_fn = get_segment(segment, original=True)

    if rlog_fn is None:
      print("failed to get segment %s" % segment)
      sys.exit(1)

    lr = LogReader(rlog_fn)
    print('injecting model into % s' % segment)
    lr = inject_model(lr, segment)

    route_name, segment_num = segment.rsplit("--", 1)
    log_fn = "%s/%s/rlog_%s.bz2" % (route_name.replace("|", "/"), segment_num, ref_commit)
    tmp_name = 'tmp_%s_%s' % (route_name, segment_num)
    save_log(tmp_name, lr)

    if not no_upload:
      upload_file(tmp_name, log_fn)
      print('uploaded %s', log_fn)
      os.remove(tmp_name)
    os.remove(rlog_fn)

  print("done")
