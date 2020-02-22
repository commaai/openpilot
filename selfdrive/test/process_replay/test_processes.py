#!/usr/bin/env python3
import argparse
import os
import requests
import sys
import tempfile

from selfdrive.test.process_replay.compare_logs import compare_logs
from selfdrive.test.process_replay.process_replay import replay_process, CONFIGS
from tools.lib.logreader import LogReader

segments = [
  ("HONDA", "0375fdf7b1ce594d|2019-06-13--08-32-25--3"), # HONDA.ACCORD
  ("HONDA", "99c94dc769b5d96e|2019-08-03--14-19-59--2"), # HONDA.CIVIC
  ("TOYOTA", "cce908f7eb8db67d|2019-08-02--15-09-51--3"), # TOYOTA.COROLLA_TSS2
  ("GM", "7ad88f53d406b787|2019-07-09--10-18-56--8"), # GM.VOLT
  ("HYUNDAI", "704b2230eb5190d6|2019-07-06--19-29-10--0"), # HYUNDAI.KIA_SORENTO
  ("CHRYSLER", "b6e1317e1bfbefa6|2019-07-06--04-05-26--5"), # CHRYSLER.JEEP_CHEROKEE
  ("SUBARU", "7873afaf022d36e2|2019-07-03--18-46-44--0"), # SUBARU.IMPREZA
  ("VW", "b0c9d2329ad1606b|2020-02-19--16-29-36--7"), # VW.GOLF
]

BASE_URL = "https://commadataci.blob.core.windows.net/openpilotci/"

def get_segment(segment_name):
  route_name, segment_num = segment_name.rsplit("--", 1)
  rlog_url = BASE_URL + "%s/%s/rlog.bz2" % (route_name.replace("|", "/"), segment_num)
  req = requests.get(rlog_url)
  assert req.status_code == 200, ("Failed to download log for %s" % segment_name)

  with tempfile.NamedTemporaryFile(delete=False, suffix=".bz2") as f:
    f.write(req.content)
    return f.name

def test_process(cfg, lr):
  # replay the log through the process and get its output
  log_msgs = replay_process(cfg, lr)

  cmp_log_fn = "%s_%s_%s.bz2" % (segment, cfg.proc_name, ref_commit)

  if not os.path.isfile(cmp_log_fn):
    req = requests.get(BASE_URL + os.path.basename(cmp_log_fn))
    assert req.status_code == 200, ("Failed to download %s" % cmp_log_fn)

    with tempfile.NamedTemporaryFile(suffix=".bz2") as f:
      f.write(req.content)
      f.flush()
      f.seek(0)
      cmp_log_msgs = list(LogReader(f.name))
  else:
    cmp_log_msgs = list(LogReader(cmp_log_fn))

  return compare_logs(cmp_log_msgs, log_msgs, cfg.ignore)

def prettyprint_diff(results):
  ret = "***** tested against commit %s *****\n" % ref_commit

  failed = False
  for segment, result in list(results.items()):
    #f.write("***** differences for segment %s *****\n" % segment)
    #print("***** results for segment %s *****" % segment)
    ret += "***** results for segment %s *****\n" % segment

    for proc, diff in list(result.items()):
      #f.write("*** process: %s ***\n" % proc)
      #print("\t%s" % proc)
      ret += "*** process: %s ***\n" % proc

      if isinstance(diff, str):
        #print("\t\t%s" % diff)
        ret += "\t\t%s" % diff

        failed = True
      elif len(diff):
        cnt = {}
        for d in diff:
          #f.write("\t%s\n" % str(d))
          #ret += "\t"

          k = str(d[1])
          cnt[k] = 1 if k not in cnt else cnt[k] + 1

        for k, v in sorted(cnt.items()):
          #print("\t\t%s: %s" % (k, v))
          ret += "\t\t%s: %s" % (k, v)
        failed = True
  return ret, failed


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="Regression test to identify changes in a process's output")

  # whitelist has precedence over blacklist in case both are defined
  parser.add_argument("--whitelist-procs", type=str, nargs="*", default=[], help="Whitelist given processes from the test (e.g. controlsd)")
  parser.add_argument("--whitelist-cars", type=str, nargs="*", default=[], help="Whitelist given cars from the test (e.g. HONDA)")


  parser.add_argument("--blacklist-procs", type=str, nargs="*", default=[], help="Blacklist given processes from the test (e.g. controlsd)")
  parser.add_argument("--blacklist-cars", type=str, nargs="*", default=[], help="Blacklist given cars from the test (e.g. HONDA)")

  args = parser.parse_args()

  cars_whitelisted = len(args.whitelist_cars) > 0
  procs_whitelisted = len(args.whitelist_procs) > 0

  process_replay_dir = os.path.dirname(os.path.abspath(__file__))
  ref_commit_fn = os.path.join(process_replay_dir, "ref_commit")

  if not os.path.isfile(ref_commit_fn):
    print("couldn't find reference commit")
    sys.exit(1)

  ref_commit = open(ref_commit_fn).read().strip()
  print("***** testing against commit %s *****" % ref_commit)

  results = {}
  for car_brand, segment in segments:
    if (cars_whitelisted and car_brand.upper() not in args.whitelist_cars) or \
        (not cars_whitelisted and car_brand.upper() in args.blacklist_cars):
      continue

    print("***** testing route segment %s *****\n" % segment)

    results[segment] = {}

    rlog_fn = get_segment(segment)
    lr = LogReader(rlog_fn)

    for cfg in CONFIGS:
      if (procs_whitelisted and cfg.proc_name not in arsgs.whitelist_procs) or \
          (not procs_whitelisted and cfg.proc_name in args.blacklist_procs):
        continue

      results[segment][cfg.proc_name] = test_process(cfg, lr)
    os.remove(rlog_fn)

  diff_txt, failed = prettyprint_diff(results)
  with open(os.path.join(process_replay_dir, "diff.txt"), "w") as f:
    f.write(diff_txt)
  print(diff_txt)

  print("TEST", "FAILED" if failed else "SUCCEEDED")

  print("\n\nTo update the reference logs for this test run:")
  print("./update_refs.py")

  sys.exit(int(failed))
