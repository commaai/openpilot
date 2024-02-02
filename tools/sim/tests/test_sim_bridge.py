import os
import subprocess
import time
import unittest

from multiprocessing import Queue

from cereal import messaging
from openpilot.common.basedir import BASEDIR

SIM_DIR = os.path.join(BASEDIR, "tools/sim")

class TestSimBridgeBase(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    if cls is TestSimBridgeBase:
      raise unittest.SkipTest("Don't run this base class, run test_metadrive_bridge.py instead")

  def setUp(self):
    self.processes = []

  def test_engage(self):
    # Startup manager and bridge.py. Check processes are running, then engage and verify.
    p_manager = subprocess.Popen("./launch_openpilot.sh", cwd=SIM_DIR)
    self.processes.append(p_manager)

    sm = messaging.SubMaster(['controlsState', 'onroadEvents', 'managerState'])
    q = Queue()
    bridge = self.create_bridge()
    p_bridge = bridge.run(q, retries=10)
    self.processes.append(p_bridge)

    max_time_per_step = 60

    # Wait for bridge to startup
    start_waiting = time.monotonic()
    while not bridge.started.value and time.monotonic() < start_waiting + max_time_per_step:
      time.sleep(0.1)
    self.assertEqual(p_bridge.exitcode, None, f"Bridge process should be running, but exited with code {p_bridge.exitcode}")

    start_time = time.monotonic()
    no_car_events_issues_once = False
    car_event_issues = []
    not_running = []
    while time.monotonic() < start_time + max_time_per_step:
      sm.update()

      not_running = [p.name for p in sm['managerState'].processes if not p.running and p.shouldBeRunning]
      car_event_issues = [event.name for event in sm['onroadEvents'] if any([event.noEntry, event.softDisable, event.immediateDisable])]

      if sm.all_alive() and len(car_event_issues) == 0 and len(not_running) == 0:
        no_car_events_issues_once = True
        break

    self.assertTrue(no_car_events_issues_once,
                    f"Failed because no messages received, or CarEvents '{car_event_issues}' or processes not running '{not_running}'")

    start_time = time.monotonic()
    min_counts_control_active = 100
    control_active = 0

    while time.monotonic() < start_time + max_time_per_step:
      sm.update()

      q.put("cruise_down")  # Try engaging

      if sm.all_alive() and sm['controlsState'].active:
        control_active += 1

        if control_active == min_counts_control_active:
          break

    self.assertEqual(min_counts_control_active, control_active, f"Simulator did not engage a minimal of {min_counts_control_active} steps was {control_active}")

  def tearDown(self):
    print("Test shutting down. CommIssues are acceptable")
    for p in reversed(self.processes):
      p.terminate()

    for p in reversed(self.processes):
      if isinstance(p, subprocess.Popen):
        p.wait(15)
      else:
        p.join(15)


if __name__ == "__main__":
  unittest.main()
