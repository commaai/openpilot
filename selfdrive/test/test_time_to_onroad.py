#!/usr/bin/env python3
import os
import time
import subprocess
import pytest
from pathlib import Path

import cereal.messaging as messaging
from common.basedir import BASEDIR
from common.timeout import Timeout
from selfdrive.updated import OVERLAY_INIT
from selfdrive.test.helpers import set_params_enabled


@pytest.mark.repeat(5)
def test_time_to_onroad():
  # launch
  set_params_enabled()
  manager_path = os.path.join(BASEDIR, "selfdrive/manager/manager.py")
  proc = subprocess.Popen(["python", manager_path])

  start_time = time.monotonic()
  sm = messaging.SubMaster(['controlsState'])
  try:
    # wait for engageability
    with Timeout(30, "timed out waiting for engageable"):
      while True:
        sm.update(1000)
        if sm['controlsState'].engageable:
          break
        time.sleep(1)
    print(f"engageable after {time.monotonic() - start_time:.2f}s")

    # once we're enageable, must be for the next few seconds
    for _ in range(500):
      sm.update(100)
      assert sm['controlsState'].engageable and sm.updated['controlsState']
  finally:
    proc.terminate()
    if proc.wait(60) is None:
      proc.kill()
