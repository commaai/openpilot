#!/usr/bin/env python3
import os
import sys
from collections import defaultdict

from opendbc.car.tests.car_diff import format_diff
from openpilot.common.git import get_commit
from openpilot.selfdrive.test.process_replay.compare_logs import compare_logs
from openpilot.selfdrive.test.process_replay.process_replay import FAKEDATA, get_process_config, replay_process
from openpilot.selfdrive.test.process_replay.test_processes import segments, get_log_data, REF_COMMIT_FN
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.url_file import URLFile

BASE_URL = "https://raw.githubusercontent.com/commaai/ci-artifacts/refs/heads/process-replay/"
CARD_CFG = get_process_config("card")
NAN_FIELDS = {'aRel', 'yvRel'}


if __name__ == "__main__":
  sys.exit(0)
