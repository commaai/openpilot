#!/usr/bin/env python3
import os
import signal
import time
import unittest

os.environ['FAKEUPLOAD'] = "1"

import selfdrive.manager as manager
from selfdrive.hardware import EON

# TODO: make eon fast
MAX_STARTUP_TIME = 30 if EON else 15
ALL_PROCESSES = manager.persistent_processes + manager.car_started_processes

class TestManager(unittest.TestCase):

  def setUp(self):
    os.environ['PASSIVE'] = '0'

  def tearDown(self):
    manager.cleanup_all_processes(None, None)

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
      manager.start_managed_process(p)
    
    time.sleep(10)

    for p in reversed(ALL_PROCESSES):
      exit_code = manager.kill_managed_process(p, retry=False)
      if not EON and (p == 'ui'or p == 'loggerd'):
        # TODO: make Qt UI exit gracefully and fix OMX encoder exiting
        continue

      # TODO: interrupted blocking read exits with 1 in cereal. use a more unique return code
      exit_codes = [0, 1]
      if p in manager.kill_processes:
        exit_codes = [-signal.SIGKILL]
      assert exit_code in exit_codes, f"{p} died with {exit_code}"



if __name__ == "__main__":
  unittest.main()
