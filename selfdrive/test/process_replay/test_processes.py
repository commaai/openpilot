#!/usr/bin/env python3
import argparse
import os
import requests
import sys
import tempfile

from selfdrive.car.car_helpers import interface_names
from selfdrive.test.process_replay.process_replay import replay_process, CONFIGS
from selfdrive.test.process_replay.compare_logs import compare_logs
from tools.lib.logreader import LogReader


INJECT_MODEL = 0

segments = [
  ("HONDA", "0375fdf7b1ce594d|2019-06-13--08-32-25--3"),      # HONDA.ACCORD
  ("HONDA", "99c94dc769b5d96e|2019-08-03--14-19-59--2"),      # HONDA.CIVIC
  ("TOYOTA", "77611a1fac303767|2020-02-29--13-29-33--3"),     # TOYOTA.COROLLA_TSS2
  ("GM", "7cc2a8365b4dd8a9|2018-12-02--12-10-44--2"),         # GM.ACADIA
  ("CHRYSLER", "b6849f5cf2c926b1|2020-02-28--07-29-48--13"),  # CHRYSLER.PACIFICA
  ("HYUNDAI", "38bfd238edecbcd7|2018-08-29--22-02-15--4"),    # HYUNDAI.SANTA_FE
  #("CHRYSLER", "b6e1317e1bfbefa6|2020-03-04--13-11-40"),   # CHRYSLER.JEEP_CHEROKEE
  ("SUBARU", "7873afaf022d36e2|2019-07-03--18-46-44--0"),     # SUBARU.IMPREZA
  ("VOLKSWAGEN", "76b83eb0245de90e|2020-03-05--19-16-05--3"), # VW.GOLF

  # Enable when port is tested and dascamOnly is no longer set
  ("NISSAN", "fbbfa6af821552b9|2020-03-03--08-09-43--0"),     # NISSAN.XTRAIL
]

# ford doesn't need to be tested until a full port is done
excluded_interfaces = ["mock", "ford"]

BASE_URL = "https://commadataci.blob.core.windows.net/openpilotci/"

# run the full test (including checks) when no args given
FULL_TEST = len(sys.argv) <= 1

def get_segment(segment_name, original=True):
  route_name, segment_num = segment_name.rsplit("--", 1)
  if original:
    rlog_url = BASE_URL + "%s/%s/rlog.bz2" % (route_name.replace("|", "/"), segment_num)
  else:
    process_replay_dir = os.path.dirname(os.path.abspath(__file__))
    model_ref_commit = open(os.path.join(process_replay_dir, "model_ref_commit")).read().strip()
    rlog_url = BASE_URL + "%s/%s/rlog_%s.bz2" % (route_name.replace("|", "/"), segment_num, model_ref_commit)
  req = requests.get(rlog_url)
  assert req.status_code == 200, ("Failed to download log for %s" % segment_name)

  with tempfile.NamedTemporaryFile(delete=False, suffix=".bz2") as f:
    f.write(req.content)
    return f.name

def test_process(cfg, lr, cmp_log_fn, ignore_fields=[], ignore_msgs=[]):
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

  log_msgs = replay_process(cfg, lr)

  # check to make sure openpilot is engaged in the route
  # TODO: update routes so enable check can run
  #       failed enable check: honda bosch, hyundai, chrysler, and subaru
  if cfg.proc_name == "controlsd" and FULL_TEST and False:
    for msg in log_msgs:
      if msg.which() == "controlsState":
        if msg.controlsState.active:
          break
    else:
      segment = cmp_log_fn.split("/")[-1].split("_")[0]
      raise Exception("Route never enabled: %s" % segment)

  return compare_logs(cmp_log_msgs, log_msgs, ignore_fields+cfg.ignore, ignore_msgs)

def format_diff(results, ref_commit):
  diff1, diff2 = "", ""
  diff2 += "***** tested against commit %s *****\n" % ref_commit

  failed = False
  for segment, result in list(results.items()):
    diff1 += "***** results for segment %s *****\n" % segment
    diff2 += "***** differences for segment %s *****\n" % segment

    for proc, diff in list(result.items()):
      diff1 += "\t%s\n" % proc
      diff2 += "*** process: %s ***\n" % proc

      if isinstance(diff, str):
        diff1 += "\t\t%s\n" % diff
        failed = True
      elif len(diff):
        cnt = {}
        for d in diff:
          diff2 += "\t%s\n" % str(d)

          k = str(d[1])
          cnt[k] = 1 if k not in cnt else cnt[k] + 1

        for k, v in sorted(cnt.items()):
          diff1 += "\t\t%s: %s\n" % (k, v)
        failed = True
  return diff1, diff2, failed

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="Regression test to identify changes in a process's output")

  # whitelist has precedence over blacklist in case both are defined
  parser.add_argument("--whitelist-procs", type=str, nargs="*", default=[],
                        help="Whitelist given processes from the test (e.g. controlsd)")
  parser.add_argument("--whitelist-cars", type=str, nargs="*", default=[],
                        help="Whitelist given cars from the test (e.g. HONDA)")
  parser.add_argument("--blacklist-procs", type=str, nargs="*", default=[],
                        help="Blacklist given processes from the test (e.g. controlsd)")
  parser.add_argument("--blacklist-cars", type=str, nargs="*", default=[],
                        help="Blacklist given cars from the test (e.g. HONDA)")
  parser.add_argument("--ignore-fields", type=str, nargs="*", default=[],
                        help="Extra fields or msgs to ignore (e.g. carState.events)")
  parser.add_argument("--ignore-msgs", type=str, nargs="*", default=[],
                        help="Msgs to ignore (e.g. carEvents)")
  args = parser.parse_args()

  cars_whitelisted = len(args.whitelist_cars) > 0
  procs_whitelisted = len(args.whitelist_procs) > 0

  process_replay_dir = os.path.dirname(os.path.abspath(__file__))
  try:
    ref_commit = open(os.path.join(process_replay_dir, "ref_commit")).read().strip()
  except:
    print("couldn't find reference commit")
    sys.exit(1)

  print("***** testing against commit %s *****" % ref_commit)

  # check to make sure all car brands are tested
  if FULL_TEST:
    tested_cars = set(c.lower() for c, _ in segments)
    untested = (set(interface_names) - set(excluded_interfaces)) - tested_cars
    assert len(untested) == 0, "Cars missing routes: %s" % (str(untested))

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
      if (procs_whitelisted and cfg.proc_name not in args.whitelist_procs) or \
          (not procs_whitelisted and cfg.proc_name in args.blacklist_procs):
        continue

      cmp_log_fn = os.path.join(process_replay_dir, "%s_%s_%s.bz2" % (segment, cfg.proc_name, ref_commit))
      results[segment][cfg.proc_name] = test_process(cfg, lr, cmp_log_fn, args.ignore_fields, args.ignore_msgs)
    os.remove(rlog_fn)

  diff1, diff2, failed = format_diff(results, ref_commit)
  with open(os.path.join(process_replay_dir, "diff.txt"), "w") as f:
    f.write(diff2)
  print(diff1)

  print("TEST", "FAILED" if failed else "SUCCEEDED")

  print("\n\nTo update the reference logs for this test run:")
  print("./update_refs.py")

  sys.exit(int(failed))
