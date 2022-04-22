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
  processes: List[subprocess.Popen]

  def setUp(self):
    self.processes = []
    # We want to make sure that carla_sim docker is still running. Skip output shell
    subprocess.run("docker rm -f carla_sim", shell=True, stderr=subprocess.PIPE, check=True)

    self.processes.append(subprocess.Popen(".././start_carla.sh"))
    time.sleep(15)

  def test_connect_with_carla(self):
    # Test connecting to Carla within 15 seconds and return no RuntimeError
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
    max_time_per_test = 20

    car_event_issues = []
    not_running = []
    while time.monotonic() < start_time + max_time_per_test:
      sm.update()

      not_running = [p.name for p in sm['managerState'].processes if not p.running and p.shouldBeRunning]
      car_event_issues = [event.name for event in sm['carEvents'] if any([event.noEntry, event.softDisable, event.immediateDisable])]

      if sm.all_alive() and len(car_event_issues) == 0 and len(not_running) == 0:
        no_car_events_issues_once = True
        break

    self.assertTrue(no_car_events_issues_once, f"Failed because of CarEvents '{car_event_issues}' or processes not running '{not_running}'")

    start_time = time.monotonic()
    control_active = 0
    while time.monotonic() < start_time + max_time_per_test:
      sm.update()

      q.put("cruise_up")  # Try engaging

      if sm.all_alive() and sm['controlsState'].active:
        control_active += 1

      if control_active > 100:
        break

    self.assertTrue(control_active, "Simulator never engaged")

  def tearDown(self):
    print("Test shutting down. CommIssues are acceptable")
    for p in reversed(self.processes):
      p.terminate()
      time.sleep(5)
    time.sleep(5)
    subprocess.run("docker rm -f carla_sim", shell=True, stderr=subprocess.PIPE, check=True)

    for p in self.processes:
      if isinstance(p, subprocess.Popen):
        p.wait()
      else:
        p.join()


if __name__ == "__main__":
  unittest.main()
