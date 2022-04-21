#!/usr/bin/env python3
import subprocess
import time
import unittest
from multiprocessing import Queue
from typing import List

from cereal import messaging
from tools.sim import bridge
from tools.sim.bridge import connect_carla_client


class TestCarlaIntegration(unittest.TestCase):
  """
  Tests need Carla simulator to run
  """
  processes: List[subprocess.Popen] = []

  def test_connect_with_carla(self):
    # Test connecting to Carla within 5 seconds and return no RuntimeError
    client = connect_carla_client()
    assert client is not None
    # Will raise an error if not connected
    client.load_world('Town04_Opt')

  def test_run_bridge(self):
    # Test bridge connect with carla and runs without any errors for 60 seconds
    test_duration = 60

    q = Queue()
    p = bridge.main(q, bridge.parse_args(['--low_quality']), keep_alive=False)
    self.processes = [p]

    time.sleep(test_duration)

    self.assertEqual(p.exitcode, None, f"Bridge process should be running, but exited with code {p.exitcode}")

  def test_engage(self):
    # Startup manager and bridge.py. Check processes are running, then engage and verify.
    p_manager = subprocess.Popen("./launch_openpilot.sh", cwd='../')
    self.processes.append(p_manager)

    sm = messaging.SubMaster(['controlsState', 'carEvents', 'managerState'])
    q = Queue()
    p_bridge = bridge.main(q, bridge.parse_args(['--low_quality']), keep_alive=False)
    self.processes.append(p_bridge)
    
    # Wait for bridge to startup
    time.sleep(10)
    self.assertEqual(p_bridge.exitcode, None, f"Bridge process should be running, but exited with code {p_bridge.exitcode}")

    start_time = time.monotonic()

    no_car_events_issues_once = False
    max_time_per_test = 10

    car_event_issues = []
    not_running = []
    while time.monotonic() < start_time + max_time_per_test:
      sm.update()
      not_running = [p.name for p in sm['managerState'].processes if not p.running and p.shouldBeRunning]
      car_event_issues = [event.name for event in sm['carEvents'] if any([event.noEntry, event.softDisable, event.immediateDisable])]

      if len(car_event_issues) == 0 and len(not_running) == 0:
        no_car_events_issues_once = True
        break

      time.sleep(0.1)
    self.assertTrue(no_car_events_issues_once, f"Failed because of CarEvents '{car_event_issues}' or processes not running '{not_running}'")

    start_time = time.monotonic()
    control_active = False
    while time.monotonic() < start_time + max_time_per_test:
      q.put("cruise_up")  # Try engaging

      sm.update()

      if sm['controlsState'].active:
        control_active = True
        break

      time.sleep(0.1)

    self.assertTrue(control_active, "Simulator never engaged")

  def tearDown(self):
    for p in self.processes:
      p.terminate()


if __name__ == "__main__":
  unittest.main()
