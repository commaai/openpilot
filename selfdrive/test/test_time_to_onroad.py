#!/usr/bin/env python3
import os
import time
import subprocess

import cereal.messaging as messaging
from openpilot.common.basedir import BASEDIR
from openpilot.common.timeout import Timeout
from openpilot.selfdrive.test.helpers import set_params_enabled


def test_time_to_onroad():
  # launch
  set_params_enabled()
  manager_path = os.path.join(BASEDIR, "selfdrive/manager/manager.py")
  proc = subprocess.Popen(["python", manager_path])

  start_time = time.monotonic()
  sm = messaging.SubMaster(['controlsState', 'deviceState', 'carEvents'])
  try:
    # wait for onroad
    with Timeout(20, "timed out waiting to go onroad"):
      while True:
        sm.update(1000)
        if sm['deviceState'].started:
          break
        time.sleep(1)

    # wait for engageability
    with Timeout(10, "timed out waiting for engageable"):
      while True:
        sm.update(1000)
        if sm['controlsState'].engageable:
          break
        time.sleep(1)
    print(f"engageable after {time.monotonic() - start_time:.2f}s")

    # once we're enageable, must be for the next few seconds
    for _ in range(500):
      sm.update(100)
      assert sm['controlsState'].engageable, f"events: {sm['carEvents']}"
  finally:
    proc.terminate()
    if proc.wait(60) is None:
      proc.kill()
