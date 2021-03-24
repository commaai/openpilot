#!/usr/bin/env python3
import os
import signal
import time
import unittest

import selfdrive.manager.manager as manager
from selfdrive.hardware import EON
from selfdrive.manager.process import DaemonProcess
from selfdrive.manager.process_config import managed_processes

os.environ['FAKEUPLOAD'] = "1"

# TODO: make eon fast
MAX_STARTUP_TIME = 30 if EON else 15
ALL_PROCESSES = [p.name for p in managed_processes.values() if (type(p) is not DaemonProcess) and (p.name not in ['updated', 'pandad'])]


class TestManager(unittest.TestCase):
  def setUp(self):
    os.environ['PASSIVE'] = '0'

  def tearDown(self):
    manager.manager_cleanup()

  def test_manager_prepare(self):
    os.environ['PREPAREONLY'] = '1'
    manager.main()

  def test_startup_time(self):
    for _ in range(10):
      start = time.monotonic()
      os.environ['PREPAREONLY'] = '1'
      manager.main()
      t = time.monotonic() - start
      assert t < MAX_STARTUP_TIME, f"startup took {t}s, expected <{MAX_STARTUP_TIME}s"

  # ensure all processes exit cleanly
  def test_clean_exit(self):
    manager.manager_prepare()

    for p in ALL_PROCESSES:
      managed_processes[p].start()

    time.sleep(30)

    for p in reversed(ALL_PROCESSES):
      exit_code = managed_processes[p].stop(retry=False)
      if (not EON and p == 'ui') or (EON and p == 'logcatd'):
        # TODO: make Qt UI exit gracefully and fix OMX encoder exiting
        continue

      # Make sure the process is actually dead
      managed_processes[p].stop()

      # TODO: interrupted blocking read exits with 1 in cereal. use a more unique return code
      exit_codes = [0, 1]
      if managed_processes[p].sigkill:
        exit_codes = [-signal.SIGKILL]
      assert exit_code in exit_codes, f"{p} died with {exit_code}"


if __name__ == "__main__":
  unittest.main()
