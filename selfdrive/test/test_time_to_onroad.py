import os
import pytest
import time
import subprocess

from cereal import car
import cereal.messaging as messaging
from openpilot.common.basedir import BASEDIR
from openpilot.common.timeout import Timeout
from openpilot.selfdrive.test.helpers import set_params_enabled

EventName = car.OnroadEvent.EventName


@pytest.mark.tici
def test_time_to_onroad():
  # launch
  set_params_enabled()
  manager_path = os.path.join(BASEDIR, "system/manager/manager.py")
  proc = subprocess.Popen(["python", manager_path])

  start_time = time.monotonic()
  sm = messaging.SubMaster(['selfdriveState', 'deviceState', 'onroadEvents'])
  try:
    # wait for onroad. timeout assumes panda is up to date
    with Timeout(10, "timed out waiting to go onroad"):
      while not sm['deviceState'].started:
        sm.update(100)

    # wait for engageability
    try:
      with Timeout(10, "timed out waiting for engageable"):
        initialized = False
        while True:
          sm.update(100)

          if sm.seen['onroadEvents'] and not any(EventName.selfdriveInitializing == e.name for e in sm['onroadEvents']):
            initialized = True

          if initialized:
            sm.update(100)
            assert sm['selfdriveState'].engageable, f"events: {sm['onroadEvents']}"
            break
    finally:
      print(f"onroad events: {sm['onroadEvents']}")
    print(f"engageable after {time.monotonic() - start_time:.2f}s")

    # once we're enageable, must stay for the next few seconds
    st = time.monotonic()
    while (time.monotonic() - st) < 10.:
      sm.update(100)
      assert sm.all_alive(), sm.alive
      assert sm['selfdriveState'].engageable, f"events: {sm['onroadEvents']}"
  finally:
    proc.terminate()
    if proc.wait(20) is None:
      proc.kill()
