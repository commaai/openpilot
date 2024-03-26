#!/usr/bin/env python3
import os
import time
import subprocess
from collections import defaultdict

import cereal.messaging as messaging
from openpilot.common.basedir import BASEDIR
from openpilot.selfdrive.test.helpers import set_params_enabled

if __name__ == "__main__":
  set_params_enabled()

  start_time = time.monotonic()
  first_times = defaultdict(float)

  socks = ['roadCameraState', 'deviceState', 'pandaStates', 'managerState', 'gyroscope', 'procLog', 'liveCalibration']
  sm = messaging.SubMaster(socks)
  first_times = dict.fromkeys(socks + ['started', 'ignition', 'manager_started'], None)

  with open(os.path.join(BASEDIR, 'prebuilt'), 'w') as f:
    f.write('\n')

  proc = subprocess.Popen(os.path.join(BASEDIR, "launch_openpilot.sh"), cwd=BASEDIR)
  try:
    while any(x is None for x in first_times.values()) and proc.poll() is None and (time.monotonic() - start_time) < 20.:
      sm.update(100)
      now = time.monotonic() - start_time

      for sock in sm.sock.keys():
        if sm.updated[sock] and first_times[sock] is None:
          first_times[sock] = now

      if first_times['started'] is None and sm['deviceState'].started:
        first_times['started'] = now
      if first_times['ignition'] is None and len(sm['pandaStates']) and sm['pandaStates'][0].ignitionLine:
        first_times['ignition'] = now

      manager_started = any(p.name == 'controlsd' and p.shouldBeRunning for p in sm['managerState'].processes)
      if first_times['manager_started'] is None and manager_started:
        first_times['manager_started'] = now
  finally:
    print("killing manager ************")
    os.system("pkill -f ./manager.py")
    proc.terminate()
    if proc.wait(20) is None:
      proc.kill()
    time.sleep(4)

    print("\n\n" + "="*10, "first seen", "="*10)
    for s, t in dict(sorted(first_times.items(), key=lambda x: x[1])).items():
      print(s.ljust(20), round(t, 2) if t is not None else None)
