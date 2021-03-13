#!/usr/bin/env python3
import os
import signal
import subprocess
import time
import unittest

import cereal.messaging as messaging
from common.basedir import BASEDIR
from common.timeout import Timeout

class TestSimulator(unittest.TestCase):

  def test_run(self):

    p_bridge = subprocess.Popen(f'{os.path.join(BASEDIR, "tools/sim/bridge.py")} --low_quality', shell=True, start_new_session=True)
    p_openpilot = subprocess.Popen(f'QT_QPA_PLATFORM=offscreen {os.path.join(BASEDIR, "tools/sim/launch_openpilot.sh")}', shell=True, start_new_session=True)

    sm = messaging.SubMaster(["controlsState"])

    with Timeout(60):
      while sm['controlsState'].upAccelCmd == 0.0:
        sm.update(0)

    time.sleep(10)

    os.killpg(os.getpgid(p_bridge.pid), signal.SIGTERM)
    os.killpg(os.getpgid(p_openpilot.pid), signal.SIGTERM)

if __name__ == "__main__":
  unittest.main()
