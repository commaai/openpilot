#!/usr/bin/env python3
import os
import pytest
import time
import subprocess

import cereal.messaging as messaging
from openpilot.common.basedir import BASEDIR
from openpilot.common.timeout import Timeout
from openpilot.selfdrive.test.helpers import set_params_enabled


@pytest.mark.tici
def test_time_to_onroad():
  # launch
  set_params_enabled()
  manager_path = os.path.join(BASEDIR, "selfdrive/manager/manager.py")
  proc = subprocess.Popen(["python", manager_path])

  start_time = time.monotonic()
  sm = messaging.SubMaster(['controlsState', 'deviceState', 'onroadEvents', 'sendcan'])
  try:
    # wait for onroad. timeout assumes panda is up to date
    with Timeout(10, "timed out waiting to go onroad"):
      while not sm['deviceState'].started:
        sm.update(100)

    # wait for engageability
    try:
      with Timeout(10, "timed out waiting for engageable"):
        sendcan_frame = None
        while True:
          sm.update(100)

          # sendcan is only sent once we're initialized
          if sm.seen['controlsState'] and sendcan_frame is None:
            sendcan_frame = sm.frame

          if sendcan_frame is not None and sm.recv_frame['sendcan'] > sendcan_frame:
            sm.update(100)
            assert sm['controlsState'].engageable, f"events: {sm['onroadEvents']}"
            break
    finally:
      print(f"onroad events: {sm['onroadEvents']}")
    print(f"engageable after {time.monotonic() - start_time:.2f}s")

    # once we're enageable, must stay for the next few seconds
    st = time.monotonic()
    while (time.monotonic() - st) < 10.:
      sm.update(100)
      assert sm.all_alive(), sm.alive
      assert sm['controlsState'].engageable, f"events: {sm['onroadEvents']}"
      assert sm['controlsState'].cumLagMs < 10.
  finally:
    proc.terminate()
    if proc.wait(20) is None:
      proc.kill()
