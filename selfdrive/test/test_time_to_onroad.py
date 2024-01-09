#!/usr/bin/env python3
import os
import pytest
import time
import subprocess
from collections import defaultdict

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

  frame_ids = defaultdict(list)
  socks = [messaging.sub_sock(s+'CameraState', conflate=False) for s in ('road', 'driver', 'wideRoad')]
  def us():
    for s in socks:
      msgs = messaging.drain_sock(s)
      for m in msgs:
        cs = getattr(m, m.which())
        frame_ids[m.which()].append(cs.frameId)

  start_time = time.monotonic()
  sm = messaging.SubMaster(['controlsState', 'deviceState', 'onroadEvents'])
  try:
    # wait for onroad
    with Timeout(20, "timed out waiting to go onroad"):
      while True:
        us()
        sm.update(0)
        if sm['deviceState'].started:
          break
        time.sleep(0.01)

    # wait for engageability
    USER_DISABLE_EVENTS = {'speedTooLow', 'reverseGear', 'wrongGear', 'doorOpen', 'seatbeltNotLatched', 'cruiseDisabled', 'espDisabled'}
    engageable = False
    st = time.monotonic()
    while (time.monotonic() - st) < 40.:
      sm.update(10)
      #evts = set([str(e.name) for e in sm['onroadEvents']])
      for e in sm['onroadEvents']:
        assert not (e.immediateDisable or e.softDisable), f"events: {sm['onroadEvents']}"

      if not engageable:
        if sm['controlsState'].engageable:
          engageable = True
        elif time.monotonic() - st > 10.:
          assert sm['controlsState'].engageable, f"events: {sm['onroadEvents']}"
      else:
        assert sm.all_checks()
        assert sm['controlsState'].engageable, f"events: {sm['onroadEvents']}"


    #for k, v in frame_ids.items():
    #  print(k, v)
    #  assert len(set(v)) == len(v)
  finally:
    proc.terminate()
    if proc.wait(60) is None:
      proc.kill()
