#!/usr/bin/env python3
import os
import subprocess
import sys
import time
import unittest

from cereal import messaging

from common.params import Params
from selfdrive.manager import manager
from selfdrive.manager.process import DaemonProcess
from selfdrive.manager.process_config import managed_processes
from tools.sim import bridge
from tools.sim.bridge import connect_carla_client


class TestCarlaIntegration(unittest.TestCase):
  """
  Tests need Carla simulator to run
  """
  subprocess = None

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

  def test_engage(self):
    # Startup manager and bridge.py. Check processes are running, then engage and verify.
    # Set environment vars to get the exact processes used by manager.py
    os.environ["PASSIVE"] = "0"
    os.environ["NOBOARD"] = "1"
    os.environ["SIMULATION"] = "1"
    os.environ["FINGERPRINT"] = "HONDA CIVIC 2016"
    os.environ["BLOCK"] = "camerad,loggerd"

    self.subprocess = subprocess.Popen("../../../selfdrive/manager/manager.py")
    sm = messaging.SubMaster(['controlsState', 'carEvents', 'managerState'])

    args = sys.argv[2:]  # Remove test arguments when executing this test
    p_bridge, _, q = bridge.main(args, keep_alive=False)

    params = Params()
    ignore_processes = manager.ignored_processes(params)+['bridge', 'webjoystick']
    all_processes = [p.name for p in managed_processes.values() if
                     (type(p) is not DaemonProcess) and p.enabled and (p.name not in ignore_processes)]
    start_time = time.time()

    no_car_events_issues_once = False
    max_time = 10
    while time.time() < start_time + max_time:
      sm.update()
      running = {p.name for p in sm['managerState'].processes if p.running}
      expected_and_not_running = [p for p in all_processes if p not in running]

      if len(sm['carEvents']) == 0 and len(expected_and_not_running) == 0:
        no_car_events_issues_once = True
        break

      time.sleep(0.1)
    self.assertTrue(no_car_events_issues_once)

    start_time = time.time()
    control_active = False
    while time.time() < start_time + max_time:
      q.put("cruise_up")  # Try engaging

      sm.update()

      if sm['controlsState'].active:
        control_active = True
        break

      time.sleep(0.1)

    self.assertTrue(control_active)

    # Set shutdown to close manager and bridge processes
    Params().put_bool("DoShutdown", True)
    # Process could already be closed
    if type(self.subprocess) is subprocess.Popen:
      self.subprocess.wait(timeout=10)
      # Closing gracefully
      self.assertEqual(self.subprocess.returncode, 0)
    p_bridge.join()

  def tearDown(self):
    print("Teardown")
    Params().put_bool("DoShutdown", True)
    manager.manager_cleanup()
    if self.subprocess:
      self.subprocess.wait(5)


if __name__ == "__main__":
  unittest.main()
