#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Any

from selfdrive.car.car_helpers import interface_names
from selfdrive.test.openpilotci import get_url
from selfdrive.test.process_replay.compare_logs import compare_logs
from selfdrive.test.process_replay.process_replay import CONFIGS, replay_process
from tools.lib.logreader import LogReader


original_segments = [
  ("HYUNDAI", "02c45f73a2e5c6e9|2021-01-01--19-08-22--1"),     # HYUNDAI.SONATA
  ("TOYOTA", "0982d79ebb0de295|2021-01-04--17-13-21--13"),     # TOYOTA.PRIUS (INDI)
  ("TOYOTA2", "0982d79ebb0de295|2021-01-03--20-03-36--6"),     # TOYOTA.RAV4  (LQR)
  ("TOYOTA3", "f7d7e3538cda1a2a|2021-08-16--08-55-34--6"),     # TOYOTA.COROLLA_TSS2
  ("HONDA", "eb140f119469d9ab|2021-06-12--10-46-24--27"),      # HONDA.CIVIC (NIDEC)
  ("HONDA2", "7d2244f34d1bbcda|2021-06-25--12-25-37--26"),     # HONDA.ACCORD (BOSCH)
  ("CHRYSLER", "4deb27de11bee626|2021-02-20--11-28-55--8"),    # CHRYSLER.PACIFICA
  ("SUBARU", "4d70bc5e608678be|2021-01-15--17-02-04--5"),      # SUBARU.IMPREZA
  ("GM", "0c58b6a25109da2b|2021-02-23--16-35-50--11"),         # GM.VOLT
  ("NISSAN", "35336926920f3571|2021-02-12--18-38-48--46"),     # NISSAN.XTRAIL
  ("VOLKSWAGEN", "de9592456ad7d144|2021-06-29--11-00-15--6"),  # VOLKSWAGEN.GOLF
  ("MAZDA", "bd6a637565e91581|2021-10-30--15-14-53--2"),       # MAZDA.CX9_2021

  # Enable when port is tested and dascamOnly is no longer set
  #("TESLA", "bb50caf5f0945ab1|2021-06-19--17-20-18--3"),      # TESLA.AP2_MODELS
]

segments = [
  ("HYUNDAI", "fakedata|2021-10-07--15-56-26--0"),
  ("TOYOTA", "fakedata|2021-10-07--15-57-47--0"),
  ("TOYOTA2", "fakedata|2021-10-07--15-59-03--0"),
  ("TOYOTA3", "fakedata|2021-10-07--15-53-21--0"),
  ("HONDA", "fakedata|2021-10-07--16-00-19--0"),
  ("HONDA2", "fakedata|2021-10-07--16-01-35--0"),
  ("CHRYSLER", "fakedata|2021-10-07--16-02-52--0"),
  ("SUBARU", "fakedata|2021-10-07--16-04-09--0"),
  ("GM", "fakedata|2021-10-07--16-05-26--0"),
  ("NISSAN", "fakedata|2021-10-07--16-09-53--0"),
  ("VOLKSWAGEN", "fakedata|2021-10-07--16-11-11--0"),
  ("MAZDA", "bd6a637565e91581|2021-10-30--15-14-53--2"),
]

# dashcamOnly makes don't need to be tested until a full port is done
excluded_interfaces = ["mock", "ford", "mazda", "tesla"]

BASE_URL = "https://commadataci.blob.core.windows.net/openpilotci/"

# run the full test (including checks) when no args given
FULL_TEST = len(sys.argv) <= 1


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
      raise Exception(f"Route never enabled: {segment}")

  try:
    return compare_logs(cmp_log_msgs, log_msgs, ignore_fields+cfg.ignore, ignore_msgs, cfg.tolerance)
  except Exception as e:
    return str(e)

def format_diff(results, ref_commit):
  diff1, diff2 = "", ""
  diff2 += f"***** tested against commit {ref_commit} *****\n"

  failed = False
  for segment, result in list(results.items()):
    diff1 += f"***** results for segment {segment} *****\n"
    diff2 += f"***** differences for segment {segment} *****\n"

    for proc, diff in list(result.items()):
      diff1 += f"\t{proc}\n"
      diff2 += f"*** process: {proc} ***\n"

      if isinstance(diff, str):
        diff1 += f"\t\t{diff}\n"
        failed = True
      elif len(diff):
        cnt = {}
        for d in diff:
          diff2 += f"\t{str(d)}\n"

          k = str(d[1])
          cnt[k] = 1 if k not in cnt else cnt[k] + 1

        for k, v in sorted(cnt.items()):
          diff1 += f"\t\t{k}: {v}\n"
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

  print(f"***** testing against commit {ref_commit} *****")

  # check to make sure all car brands are tested
  if FULL_TEST:
    tested_cars = {c.lower() for c, _ in segments}
    untested = (set(interface_names) - set(excluded_interfaces)) - tested_cars
    assert len(untested) == 0, f"Cars missing routes: {str(untested)}"

  results: Any = {}
  for car_brand, segment in segments:
    if (cars_whitelisted and car_brand.upper() not in args.whitelist_cars) or \
       (not cars_whitelisted and car_brand.upper() in args.blacklist_cars):
      continue

    print(f"***** testing route segment {segment} *****\n")

    results[segment] = {}

    r, n = segment.rsplit("--", 1)
    lr = LogReader(get_url(r, n))

    for cfg in CONFIGS:
      if (procs_whitelisted and cfg.proc_name not in args.whitelist_procs) or \
         (not procs_whitelisted and cfg.proc_name in args.blacklist_procs):
        continue

      cmp_log_fn = os.path.join(process_replay_dir, f"{segment}_{cfg.proc_name}_{ref_commit}.bz2")
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
