import os
import subprocess
import time
import unittest

from multiprocessing import Queue, Value

from cereal import messaging
from openpilot.common.basedir import BASEDIR

SIM_DIR = os.path.join(BASEDIR, "tools/sim")


class TestSimBridgeBase(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    if cls is TestSimBridgeBase:
      raise unittest.SkipTest("Don't run this base class, run test_metadrive_bridge.py instead")
    TestSimBridgeBase.drive_time = cls.drive_time if cls.drive_time else 60

  def setUp(self):
    self.processes = []

  def test_engage(self):
    # Startup manager and bridge.py. Check processes are running, then engage and verify.
    p_manager = subprocess.Popen("./launch_openpilot.sh", cwd=SIM_DIR)
    sm = messaging.SubMaster(['controlsState', 'onroadEvents', 'managerState'])
    self.processes.append(p_manager)

    q = Queue()
    bridge = self.create_bridge()
    bridge.started = Value('b', False)
    p_bridge = bridge.run(q, retries=10)
    self.processes.append(p_bridge)

    # Wait for bridge to startup
    max_time_per_step = 60

    start_waiting = time.monotonic()
    while not bridge.started and time.monotonic() < start_waiting + max_time_per_step:
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

    # Drive course
    q.put("reset")
    start_time = time.monotonic()
    user_disengage_once = False
    disengage_events = ('stockAeb', 'fcw', 'ldw')

    while time.monotonic() < start_time + TestSimBridgeBase.drive_time:
      sm.update()
      onroadEventNames = [e.name for e in sm['onroadEvents']]
      if any(e in onroadEventNames for e in disengage_events):
        user_disengage_once = True
        break
    self.assertFalse(user_disengage_once, "Failed because user has to disengage")

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
