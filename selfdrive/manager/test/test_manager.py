#!/usr/bin/env python3
import os
import pytest
import signal
import time
import unittest

from parameterized import parameterized

from cereal import car
from openpilot.common.params import Params
import openpilot.selfdrive.manager.manager as manager
from openpilot.selfdrive.manager.process import ensure_running
from openpilot.selfdrive.manager.process_config import managed_processes
from openpilot.system.hardware import HARDWARE

os.environ['FAKEUPLOAD'] = "1"

MAX_STARTUP_TIME = 3
BLACKLIST_PROCS = ['manage_athenad', 'pandad', 'pigeond']


@pytest.mark.tici
class TestManager(unittest.TestCase):
  def setUp(self):
    HARDWARE.set_power_save(False)

    # ensure clean CarParams
    params = Params()
    params.clear_all()

  def tearDown(self):
    manager.manager_cleanup()

  def test_manager_prepare(self):
    os.environ['PREPAREONLY'] = '1'
    manager.main()

  def test_blacklisted_procs(self):
    # TODO: ensure there are blacklisted procs until we have a dedicated test
    self.assertTrue(len(BLACKLIST_PROCS), "No blacklisted procs to test not_run")

  @parameterized.expand([(i,) for i in range(10)])
  def test_startup_time(self, index):
    start = time.monotonic()
    os.environ['PREPAREONLY'] = '1'
    manager.main()
    t = time.monotonic() - start
    assert t < MAX_STARTUP_TIME, f"startup took {t}s, expected <{MAX_STARTUP_TIME}s"

  @unittest.skip("this test is flaky the way it's currently written, should be moved to test_onroad")
  def test_clean_exit(self):
    """
      Ensure all processes exit cleanly when stopped.
    """
    HARDWARE.set_power_save(False)
    manager.manager_init()

    CP = car.CarParams.new_message()
    procs = ensure_running(managed_processes.values(), True, Params(), CP, not_run=BLACKLIST_PROCS)

    time.sleep(10)

    for p in procs:
      with self.subTest(proc=p.name):
        state = p.get_process_state_msg()
        self.assertTrue(state.running, f"{p.name} not running")
        exit_code = p.stop(retry=False)

        self.assertNotIn(p.name, BLACKLIST_PROCS, f"{p.name} was started")

        self.assertTrue(exit_code is not None, f"{p.name} failed to exit")

        # TODO: interrupted blocking read exits with 1 in cereal. use a more unique return code
        exit_codes = [0, 1]
        if p.sigkill:
          exit_codes = [-signal.SIGKILL]
        self.assertIn(exit_code, exit_codes, f"{p.name} died with {exit_code}")


if __name__ == "__main__":
  unittest.main()
