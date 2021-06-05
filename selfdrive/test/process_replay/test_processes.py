#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Any

from selfdrive.car.car_helpers import interface_names
from selfdrive.test.process_replay.compare_logs import compare_logs
from selfdrive.test.process_replay.process_replay import CONFIGS, replay_process
from tools.lib.logreader import LogReader

INJECT_MODEL = 0

segments = [
  ("HYUNDAI", "02c45f73a2e5c6e9|2021-01-01--19-08-22--1"),     # HYUNDAI.SONATA
  ("TOYOTA", "0982d79ebb0de295|2021-01-04--17-13-21--13"),     # TOYOTA.PRIUS (INDI)
  ("TOYOTA2", "0982d79ebb0de295|2021-01-03--20-03-36--6"),     # TOYOTA.RAV4  (LQR)
  ("HONDA", "0982d79ebb0de295|2021-01-08--10-13-10--6"),       # HONDA.CIVIC (NIDEC)
  ("HONDA2", "a8e8bf6a3864361b|2021-01-04--03-01-18--2"),      # HONDA.ACCORD (BOSCH)
  ("CHRYSLER", "52d86230ee29aa84|2021-01-10--17-16-34--30"),   # CHRYSLER.PACIFICA
  ("SUBARU", "4d70bc5e608678be|2021-01-15--17-02-04--5"),      # SUBARU.IMPREZA
  ("GM", "ae3ed0eb20960a20|2021-01-15--15-04-06--8"),          # GM.VOLT
  ("NISSAN", "e4d79cf6b8b19a0d|2021-01-17--14-48-08--7"),      # NISSAN.XTRAIL
  ("VOLKSWAGEN", "ef895f46af5fd73f|2021-05-22--14-06-35--6"),  # VW.AUDI_A3_MK3

  # Enable when port is tested and dascamOnly is no longer set
  #("MAZDA", "32a319f057902bb3|2020-04-27--15-18-58--2"),      # MAZDA.CX5
]

# dashcamOnly makes don't need to be tested until a full port is done
excluded_interfaces = ["mock", "ford", "mazda"]

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

  return rlog_url


def test_process(cfg, lr, cmp_log_fn, ignore_fields=None, ignore_msgs=None):
  if ignore_fields is None:
    ignore_fields = []
  if ignore_msgs is None:
    ignore_msgs = []

  cmp_log_path = cmp_log_fn if os.path.exists(cmp_log_fn) else BASE_URL + os.path.basename(cmp_log_fn)
  cmp_log_msgs = list(LogReader(cmp_log_path))

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

  try:
    return compare_logs(cmp_log_msgs, log_msgs, ignore_fields+cfg.ignore, ignore_msgs, cfg.tolerance)
  except Exception as e:
    return str(e)

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
  except FileNotFoundError:
    print("couldn't find reference commit")
    sys.exit(1)

  print("***** testing against commit %s *****" % ref_commit)

  # check to make sure all car brands are tested
  if FULL_TEST:
    tested_cars = set(c.lower() for c, _ in segments)
    untested = (set(interface_names) - set(excluded_interfaces)) - tested_cars
    assert len(untested) == 0, "Cars missing routes: %s" % (str(untested))

  results: Any = {}
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

  diff1, diff2, failed = format_diff(results, ref_commit)
  with open(os.path.join(process_replay_dir, "diff.txt"), "w") as f:
    f.write(diff2)
  print(diff1)

  if failed:
    print("TEST FAILED")
    print("\n\nTo update the reference logs for this test run:")
    print("./update_refs.py")
  else:
    print("TEST SUCCEEDED")

  sys.exit(int(failed))
