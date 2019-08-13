#!/usr/bin/env python2
import os
import requests
import sys
import tempfile

from selfdrive.test.tests.process_replay.compare_logs import compare_logs
from selfdrive.test.tests.process_replay.process_replay import replay_process, CONFIGS
from tools.lib.logreader import LogReader

segments = [
  "0375fdf7b1ce594d|2019-06-13--08-32-25--3", # HONDA.ACCORD
  "99c94dc769b5d96e|2019-08-03--14-19-59--2", # HONDA.CIVIC
  "cce908f7eb8db67d|2019-08-02--15-09-51--3", # TOYOTA.COROLLA_TSS2
  "7ad88f53d406b787|2019-07-09--10-18-56--8", # GM.VOLT
  "704b2230eb5190d6|2019-07-06--19-29-10--0", # HYUNDAI.KIA_SORENTO
  "b6e1317e1bfbefa6|2019-07-06--04-05-26--5", # CHRYSLER.JEEP_CHEROKEE
  "7873afaf022d36e2|2019-07-03--18-46-44--0", # SUBARU.IMPREZA
]

def get_segment(segment_name):
  route_name, segment_num = segment_name.rsplit("--", 1)
  rlog_url = "https://commadataci.blob.core.windows.net/openpilotci/%s/%s/rlog.bz2" \
                % (route_name.replace("|", "/"), segment_num)
  r = requests.get(rlog_url)
  if r.status_code != 200:
    return None

  with tempfile.NamedTemporaryFile(delete=False, suffix=".bz2") as f:
    f.write(r.content)
    return f.name

if __name__ == "__main__":

  process_replay_dir = os.path.dirname(os.path.abspath(__file__))
  ref_commit_fn = os.path.join(process_replay_dir, "ref_commit")

  if not os.path.isfile(ref_commit_fn):
    print "couldn't find reference commit"
    sys.exit(1)

  ref_commit = open(ref_commit_fn).read().strip()
  print "***** testing against commit %s *****" % ref_commit

  results = {}
  for segment in segments:
    print "***** testing route segment %s *****\n" % segment

    results[segment] = {}

    rlog_fn = get_segment(segment)

    if rlog_fn is None:
      print "failed to get segment %s" % segment
      sys.exit(1)

    lr = LogReader(rlog_fn)

    for cfg in CONFIGS:
      log_msgs = replay_process(cfg, lr)

      log_fn = os.path.join(process_replay_dir, "%s_%s_%s.bz2" % (segment, cfg.proc_name, ref_commit))

      if not os.path.isfile(log_fn):
        url = "https://commadataci.blob.core.windows.net/openpilotci/"
        req = requests.get(url + os.path.basename(log_fn))
        if req.status_code != 200:
          results[segment][cfg.proc_name] = "failed to download comparison log"
          continue

        with tempfile.NamedTemporaryFile(suffix=".bz2") as f:
          f.write(req.content)
          f.flush()
          f.seek(0)
          cmp_log_msgs = list(LogReader(f.name))
      else:
        cmp_log_msgs = list(LogReader(log_fn))

      diff = compare_logs(cmp_log_msgs, log_msgs, cfg.ignore)
      results[segment][cfg.proc_name] = diff
    os.remove(rlog_fn)

  failed = False
  with open(os.path.join(process_replay_dir, "diff.txt"), "w") as f:
    f.write("***** tested against commit %s *****\n" % ref_commit)

    for segment, result in results.items():
      f.write("***** differences for segment %s *****\n" % segment)
      print "***** results for segment %s *****" % segment

      for proc, diff in result.items():
        f.write("*** process: %s ***\n" % proc)
        print "\t%s" % proc

        if isinstance(diff, str):
          print "\t\t%s" % diff
          failed = True
        elif len(diff):
          cnt = {}
          for d in diff:
            f.write("\t%s\n" % str(d))

            k = str(d[1])
            cnt[k] = 1 if k not in cnt else cnt[k] + 1

          for k, v in sorted(cnt.items()):
            print "\t\t%s: %s" % (k, v)
          failed = True

    if failed:
      print "TEST FAILED"
    else:
      print "TEST SUCCEEDED"

  print "\n\nTo update the reference logs for this test run:"
  print "./update_refs.py"

  sys.exit(int(failed))
