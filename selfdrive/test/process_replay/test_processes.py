#!/usr/bin/env python3
import argparse
import concurrent.futures
import os
import sys
from collections import defaultdict
from tqdm import tqdm
from typing import Any

from opendbc.car.car_helpers import interface_names
from openpilot.common.git import get_commit
from openpilot.tools.lib.openpilotci import get_url, upload_file
from openpilot.selfdrive.test.process_replay.compare_logs import compare_logs, format_diff
from openpilot.selfdrive.test.process_replay.process_replay import CONFIGS, PROC_REPLAY_DIR, FAKEDATA, replay_process, \
                                                                   check_most_messages_valid
from openpilot.tools.lib.filereader import FileReader
from openpilot.tools.lib.logreader import LogReader, save_log

source_segments = [
  ("BODY", "937ccb7243511b65|2022-05-24--16-03-09--1"),        # COMMA.COMMA_BODY
  ("HYUNDAI", "02c45f73a2e5c6e9|2021-01-01--19-08-22--1"),     # HYUNDAI.HYUNDAI_SONATA
  ("HYUNDAI2", "d545129f3ca90f28|2022-11-07--20-43-08--3"),    # HYUNDAI.HYUNDAI_KIA_EV6 (+ QCOM GPS)
  ("TOYOTA", "0982d79ebb0de295|2021-01-04--17-13-21--13"),     # TOYOTA.TOYOTA_PRIUS
  ("TOYOTA2", "0982d79ebb0de295|2021-01-03--20-03-36--6"),     # TOYOTA.TOYOTA_RAV4
  ("TOYOTA3", "8011d605be1cbb77|000000cc--8e8d8ec716--6"),     # TOYOTA.TOYOTA_COROLLA_TSS2
  ("HONDA", "eb140f119469d9ab|2021-06-12--10-46-24--27"),      # HONDA.HONDA_CIVIC (NIDEC)
  ("HONDA2", "7d2244f34d1bbcda|2021-06-25--12-25-37--26"),     # HONDA.HONDA_ACCORD (BOSCH)
  ("CHRYSLER", "4deb27de11bee626|2021-02-20--11-28-55--8"),    # CHRYSLER.CHRYSLER_PACIFICA_2018_HYBRID
  ("RAM", "17fc16d840fe9d21|2023-04-26--13-28-44--5"),         # CHRYSLER.RAM_1500_5TH_GEN
  ("SUBARU", "341dccd5359e3c97|2022-09-12--10-35-33--3"),      # SUBARU.SUBARU_OUTBACK
  ("GM", "376bf99325883932|2022-10-27--13-41-22--1"),         # GM.CHEVROLET_BOLT_EUV
  ("NISSAN", "35336926920f3571|2021-02-12--18-38-48--46"),     # NISSAN.NISSAN_XTRAIL
  ("VOLKSWAGEN", "de9592456ad7d144|2021-06-29--11-00-15--6"),  # VOLKSWAGEN.VOLKSWAGEN_GOLF
  # FIXME the sensor timings are bad in mazda segment, we're not fully testing it, but it should be replaced
  ("MAZDA", "bd6a637565e91581|2021-10-30--15-14-53--4"),       # MAZDA.MAZDA_CX9_2021
  ("FORD", "54827bf84c38b14f|2023-01-26--21-59-07--4"),        # FORD.FORD_BRONCO_SPORT_MK1
  ("RIVIAN", "bc095dc92e101734|000000db--ee9fe46e57--1"),      # RIVIAN.RIVIAN_R1_GEN1
  ("TESLA", "2c912ca5de3b1ee9|0000025d--6eb6bcbca4--4"),       # TESLA.TESLA_MODEL_Y

  # Enable when port is tested and dashcamOnly is no longer set
  #("VOLKSWAGEN2", "3cfdec54aa035f3f|2022-07-19--23-45-10--2"),  # VOLKSWAGEN.VOLKSWAGEN_PASSAT_NMS
]

segments = [
  ("BODY", "regen2F3C7259F1B|2025-04-08--23-00-23--0"),
  ("HYUNDAI", "regenAA0FC4ED71E|2025-04-08--22-57-50--0"),
  ("HYUNDAI2", "regenAFB9780D823|2025-04-08--23-00-34--0"),
  ("TOYOTA", "regen218A4DCFAA1|2025-04-08--22-57-51--0"),
  ("TOYOTA2", "regen107352E20EB|2025-04-08--22-57-46--0"),
  ("TOYOTA3", "regen1455E3B4BDF|2025-04-09--03-26-06--0"),
  ("HONDA", "regenB328FF8BA0A|2025-04-08--22-57-45--0"),
  ("HONDA2", "regen6170C8C9A35|2025-04-08--22-57-46--0"),
  ("CHRYSLER", "regen5B28FC2A437|2025-04-08--23-04-24--0"),
  ("RAM", "regenBF81EA96E08|2025-04-08--23-06-54--0"),
  ("SUBARU", "regen7366F13F6A1|2025-04-08--23-07-07--0"),
  ("GM", "regen1271097D038|2025-04-09--03-26-00--0"),
  ("NISSAN", "regen15D60604EAB|2025-04-08--23-06-59--0"),
  ("VOLKSWAGEN", "regen0F2F06C9539|2025-04-08--23-06-56--0"),
  ("MAZDA", "regenACF84CCF482|2024-08-30--03-21-55--0"),
  ("FORD", "regen755D8CB1E1F|2025-04-08--23-13-43--0"),
  ("RIVIAN", "regen5FCAC896BBE|2025-04-08--23-13-35--0"),
  ("TESLA", "2c912ca5de3b1ee9|0000025d--6eb6bcbca4--4"),
]

# dashcamOnly makes don't need to be tested until a full port is done
excluded_interfaces = ["mock", "tesla"]

BASE_URL = "https://commadataci.blob.core.windows.net/openpilotci/"
REF_COMMIT_FN = os.path.join(PROC_REPLAY_DIR, "ref_commit")
EXCLUDED_PROCS = {"modeld", "dmonitoringmodeld"}


def run_test_process(data):
  segment, cfg, args, cur_log_fn, ref_log_path, lr_dat = data
  res = None
  if not args.upload_only:
    lr = LogReader.from_bytes(lr_dat)
    res, log_msgs = test_process(cfg, lr, segment, ref_log_path, cur_log_fn, args.ignore_fields, args.ignore_msgs)
    # save logs so we can upload when updating refs
    save_log(cur_log_fn, log_msgs)

  if args.update_refs or args.upload_only:
    print(f'Uploading: {os.path.basename(cur_log_fn)}')
    assert os.path.exists(cur_log_fn), f"Cannot find log to upload: {cur_log_fn}"
    upload_file(cur_log_fn, os.path.basename(cur_log_fn))
    os.remove(cur_log_fn)
  return (segment, cfg.proc_name, res)


def get_log_data(segment):
  r, n = segment.rsplit("--", 1)
  with FileReader(get_url(r, n, "rlog.zst")) as f:
    return (segment, f.read())


def test_process(cfg, lr, segment, ref_log_path, new_log_path, ignore_fields=None, ignore_msgs=None):
  if ignore_fields is None:
    ignore_fields = []
  if ignore_msgs is None:
    ignore_msgs = []

  ref_log_msgs = list(LogReader(ref_log_path))

  try:
    log_msgs = replay_process(cfg, lr, disable_progress=True)
  except Exception as e:
    raise Exception("failed on segment: " + segment) from e

  if not check_most_messages_valid(log_msgs):
    return f"Route did not have enough valid messages: {new_log_path}", log_msgs

  # skip this check if the segment is using qcom gps
  if cfg.proc_name != 'ubloxd' or any(m.which() in cfg.pubs for m in lr):
    seen_msgs = {m.which() for m in log_msgs}
    expected_msgs = set(cfg.subs)
    if seen_msgs != expected_msgs:
      return f"Expected messages: {expected_msgs}, but got: {seen_msgs}", log_msgs

  try:
    return compare_logs(ref_log_msgs, log_msgs, ignore_fields + cfg.ignore, ignore_msgs, cfg.tolerance), log_msgs
  except Exception as e:
    return str(e), log_msgs


if __name__ == "__main__":
  all_cars = {car for car, _ in segments}
  all_procs = {cfg.proc_name for cfg in CONFIGS if cfg.proc_name not in EXCLUDED_PROCS}

  cpu_count = os.cpu_count() or 1

  parser = argparse.ArgumentParser(description="Regression test to identify changes in a process's output")
  parser.add_argument("--whitelist-procs", type=str, nargs="*", default=all_procs,
                      help="Whitelist given processes from the test (e.g. controlsd)")
  parser.add_argument("--whitelist-cars", type=str, nargs="*", default=all_cars,
                      help="Whitelist given cars from the test (e.g. HONDA)")
  parser.add_argument("--blacklist-procs", type=str, nargs="*", default=[],
                      help="Blacklist given processes from the test (e.g. controlsd)")
  parser.add_argument("--blacklist-cars", type=str, nargs="*", default=[],
                      help="Blacklist given cars from the test (e.g. HONDA)")
  parser.add_argument("--ignore-fields", type=str, nargs="*", default=[],
                      help="Extra fields or msgs to ignore (e.g. driverMonitoringState.events)")
  parser.add_argument("--ignore-msgs", type=str, nargs="*", default=[],
                      help="Msgs to ignore (e.g. carEvents)")
  parser.add_argument("--update-refs", action="store_true",
                      help="Updates reference logs using current commit")
  parser.add_argument("--upload-only", action="store_true",
                      help="Skips testing processes and uploads logs from previous test run")
  parser.add_argument("-j", "--jobs", type=int, default=max(cpu_count - 2, 1),
                      help="Max amount of parallel jobs")
  args = parser.parse_args()

  tested_procs = set(args.whitelist_procs) - set(args.blacklist_procs)
  tested_cars = set(args.whitelist_cars) - set(args.blacklist_cars)
  tested_cars = {c.upper() for c in tested_cars}

  full_test = (tested_procs == all_procs) and (tested_cars == all_cars) and all(len(x) == 0 for x in (args.ignore_fields, args.ignore_msgs))
  upload = args.update_refs or args.upload_only
  os.makedirs(os.path.dirname(FAKEDATA), exist_ok=True)

  if upload:
    assert full_test, "Need to run full test when updating refs"

  try:
    with open(REF_COMMIT_FN) as f:
      ref_commit = f.read().strip()
  except FileNotFoundError:
    print("Couldn't find reference commit")
    sys.exit(1)

  cur_commit = get_commit()
  if not cur_commit:
    raise Exception("Couldn't get current commit")

  print(f"***** testing against commit {ref_commit} *****")

  # check to make sure all car brands are tested
  if full_test:
    untested = (set(interface_names) - set(excluded_interfaces)) - {c.lower() for c in tested_cars}
    assert len(untested) == 0, f"Cars missing routes: {str(untested)}"

  log_paths: defaultdict[str, dict[str, dict[str, str]]] = defaultdict(lambda: defaultdict(dict))
  with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
    if not args.upload_only:
      download_segments = [seg for car, seg in segments if car in tested_cars]
      log_data: dict[str, LogReader] = {}
      p1 = pool.map(get_log_data, download_segments)
      for segment, lr in tqdm(p1, desc="Getting Logs", total=len(download_segments)):
        log_data[segment] = lr

    pool_args: Any = []
    for car_brand, segment in segments:
      if car_brand not in tested_cars:
        continue

      for cfg in CONFIGS:
        if cfg.proc_name not in tested_procs:
          continue

        # to speed things up, we only test all segments on card
        if cfg.proc_name != 'card' and car_brand not in ('HYUNDAI', 'TOYOTA', 'HONDA', 'SUBARU', 'FORD', 'RIVIAN', 'TESLA'):
          continue

        cur_log_fn = os.path.join(FAKEDATA, f"{segment}_{cfg.proc_name}_{cur_commit}.zst")
        if args.update_refs:  # reference logs will not exist if routes were just regenerated
          ref_log_path = get_url(*segment.rsplit("--", 1,), "rlog.zst")
        else:
          ref_log_fn = os.path.join(FAKEDATA, f"{segment}_{cfg.proc_name}_{ref_commit}.zst")
          ref_log_path = ref_log_fn if os.path.exists(ref_log_fn) else BASE_URL + os.path.basename(ref_log_fn)

        dat = None if args.upload_only else log_data[segment]
        pool_args.append((segment, cfg, args, cur_log_fn, ref_log_path, dat))

        log_paths[segment][cfg.proc_name]['ref'] = ref_log_path
        log_paths[segment][cfg.proc_name]['new'] = cur_log_fn

    results: Any = defaultdict(dict)
    p2 = pool.map(run_test_process, pool_args)
    for (segment, proc, result) in tqdm(p2, desc="Running Tests", total=len(pool_args)):
      if not args.upload_only:
        results[segment][proc] = result

  diff_short, diff_long, failed = format_diff(results, log_paths, ref_commit)
  if not upload:
    with open(os.path.join(PROC_REPLAY_DIR, "diff.txt"), "w") as f:
      f.write(diff_long)
    print(diff_short)

    if failed:
      print("TEST FAILED")
      print("\n\nTo push the new reference logs for this commit run:")
      print("./test_processes.py --upload-only")
    else:
      print("TEST SUCCEEDED")

  else:
    with open(REF_COMMIT_FN, "w") as f:
      f.write(cur_commit)
    print(f"\n\nUpdated reference logs for commit: {cur_commit}")

  sys.exit(int(failed))
