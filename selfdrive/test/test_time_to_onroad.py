#!/usr/bin/env python3
import os
import time
import signal
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
  # setup
  set_params_enabled()
  OVERLAY_INIT.unlink(missing_ok=True)
  Path(os.path.join(BASEDIR, "prebuilt")).touch()

  # launch
  proc = subprocess.Popen(f"{BASEDIR}/launch_openpilot.sh", cwd=BASEDIR, start_new_session=True)
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

    # once we're enageable, must be for the next 5s
    for _ in range(500):
      sm.update(100)
      assert sm['controlsState'].engageable and sm.updated['controlsState']
  finally:
    pgrp = os.getpgid(proc.pid)
    os.killpg(pgrp, signal.SIGTERM)
    if proc.wait(60) is None:
      proc.kill()
