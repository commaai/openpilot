#!/usr/bin/env python3
import os
import subprocess
import sys
import time
import unittest

import psutil

from common.params_pyx import Params # pylint: disable=no-name-in-module
from selfdrive.manager import manager
from selfdrive.manager.process import DaemonProcess
from selfdrive.manager.process_config import managed_processes
from tools.sim import bridge
from tools.sim.bridge import connect_carla_client


class TestCarlaIntegration(unittest.TestCase):
  """
  Tests need Carla simulator to run
  """
  def test_connect_with_carla(self):
    # Test connecting to Carla within 5 seconds and return no RuntimeError
    client = connect_carla_client()
    assert client is not None
    # Will raise an error if not connected
    client.load_world('Town04_Opt')

  def test_run_bridge(self):
    # Test bridge connect with carla and runs without any errors for 60 seconds
    test_duration = 60

    params = Params()
    args = sys.argv[2:]  # Remove test arguments when executing this test
    params.put_bool("DoShutdown", False)

    p = bridge.main(args, keep_alive=False)[0]
    time.sleep(test_duration)
    params.put_bool("DoShutdown", True)

    p.join()
    # Assert no exceptions
    self.assertEqual(p.exitcode, 0)

  def assert_processes_running(self, expected_p):
    running = {p: False for p in expected_p}
    processes = psutil.process_iter()
    for p in expected_p:
      name = managed_processes[p].get_process_state_msg().name
      for proc in processes:
        if proc.name().endswith(name):
          running[name] = True
          break

    not_running = [key for (key, val) in running.items() if not val]
    self.assertEqual(len(not_running), 0, f"Some processes are not running {not_running}")

  def test_manager_and_bridge(self):
    # test manager.py processes and bridge.py to run correctly for 50 seconds
    startup_time = 10
    test_intervals_5sec = 10

    # Set params for simulation to be used for ignored_processes
    os.environ["PASSIVE"] = "0"
    os.environ["NOBOARD"] = "1"
    os.environ["SIMULATION"] = "1"
    os.environ["FINGERPRINT"] = "HONDA CIVIC 2016"
    os.environ["BLOCK"] = "camerad,loggerd"

    # Start manager and bridge
    p = subprocess.Popen("../../../selfdrive/manager/manager.py")

    args = sys.argv[2:]  # Remove test arguments when executing this test
    p_bridge = bridge.main(args, keep_alive=False)[0]

    time.sleep(startup_time)

    params = Params()
    ignore_processes = manager.ignored_processes(params)
    all_processes = [p.name for p in managed_processes.values() if
                     (type(p) is not DaemonProcess) and p.enabled and (p.name not in ignore_processes)]

    # Test for 50 seconds
    for _ in range(test_intervals_5sec):
      self.assert_processes_running(all_processes)
      time.sleep(5)

    # Set shutdown to close manager and bridge processes
    params.put_bool("DoShutdown", True)
    # Process could already be closed
    if type(p) is subprocess.Popen:
      p.wait(timeout=10)
      # Closing gracefully
      self.assertEqual(p.returncode, 0)
    p_bridge.join()

  def tearDown(self):
    print("Teardown")
    Params().put_bool("DoShutdown", True)
    manager.manager_cleanup()


if __name__ == "__main__":
  unittest.main()
