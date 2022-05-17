#!/usr/bin/env python3
import argparse
import multiprocessing.pool
import os
import pytest
import sys
from typing import Any, Dict

from selfdrive.car.car_helpers import interface_names
from selfdrive.test.openpilotci import get_url, upload_file
from selfdrive.test.process_replay.compare_logs import compare_logs, save_log
from selfdrive.test.process_replay.process_replay import CONFIGS, PROC_REPLAY_DIR, FAKEDATA, check_enabled, replay_process
from selfdrive.version import get_commit
from tools.lib.logreader import LogReader


original_segments = [
  ("BODY", "bd6a637565e91581|2022-04-04--22-05-08--0"),        # COMMA.BODY
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
  ("BODY", "bd6a637565e91581|2022-04-04--22-05-08--0"),
  ("HYUNDAI", "fakedata|2022-01-20--17-49-04--0"),
  ("TOYOTA", "fakedata|2022-04-29--15-57-12--0"),
  ("TOYOTA2", "fakedata|2022-04-29--16-08-01--0"),
  ("TOYOTA3", "fakedata|2022-04-29--16-17-39--0"),
  ("HONDA", "fakedata|2022-01-20--17-56-40--0"),
  ("HONDA2", "fakedata|2022-04-29--16-31-55--0"),
  ("CHRYSLER", "fakedata|2022-01-20--18-00-11--0"),
  ("SUBARU", "fakedata|2022-01-20--18-01-57--0"),
  ("GM", "fakedata|2022-01-20--18-03-41--0"),
  ("NISSAN", "fakedata|2022-01-20--18-05-29--0"),
  ("VOLKSWAGEN", "fakedata|2022-01-20--18-07-15--0"),
  ("MAZDA", "fakedata|2022-01-20--18-09-32--0"),
]

# dashcamOnly makes don't need to be tested until a full port is done
excluded_interfaces = ["mock", "ford", "mazda", "tesla"]

BASE_URL = "https://commadataci.blob.core.windows.net/openpilotci/"
REF_COMMIT_FN = os.path.join(PROC_REPLAY_DIR, "ref_commit")

def do_test_process(cfg, lr, ref_log_fn, ignore_fields=None, ignore_msgs=None):
  if ignore_fields is None:
    ignore_fields = []
  if ignore_msgs is None:
    ignore_msgs = []

  ref_log_path = ref_log_fn if os.path.exists(ref_log_fn) else BASE_URL + os.path.basename(ref_log_fn)
  ref_log_msgs = list(LogReader(ref_log_path))

  log_msgs = replay_process(cfg, lr)

  # check to make sure openpilot is engaged in the route
  if cfg.proc_name == "controlsd":
    if not check_enabled(log_msgs):
      segment = ref_log_fn.split("/")[-1].split("_")[0]
      raise Exception(f"Route never enabled: {segment}")

  try:
    return compare_logs(ref_log_msgs, log_msgs, ignore_fields + cfg.ignore, ignore_msgs, cfg.tolerance), log_msgs
  except Exception as e:
    return str(e), log_msgs

# Build pool args
cur_commit = get_commit()
ref_commit = open(REF_COMMIT_FN).read().strip()
pool_args: Any = []
for car_brand, segment in segments:
  for cfg in CONFIGS:
    cur_log_fn = os.path.join(FAKEDATA, f"{segment}_{cfg.proc_name}_{cur_commit}.bz2")
    pool_args.append((segment, cfg, cur_log_fn, ref_commit))

# Get CMD args
@pytest.fixture(scope="session")
def args(pytestconfig):
  args = pytestconfig.option
  return args

# Do tests
@pytest.mark.parametrize("data", pool_args)
def test_process(data, args):
  segment, cfg, cur_log_fn, ref_commit = data
  r, n = segment.rsplit("--", 1)
  lr = LogReader(get_url(r, n))
  ref_log_fn = os.path.join(PROC_REPLAY_DIR, f"{segment}_{cfg.proc_name}_{ref_commit}.bz2")
  res, log_msgs = do_test_process(cfg, lr, ref_log_fn, args.ignore_fields, args.ignore_msgs)
  prin
  if isinstance(res, str) or len(res):
    print("FAIL", segment, res)
    assert False
  assert True
