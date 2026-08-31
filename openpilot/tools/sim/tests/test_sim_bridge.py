import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import unittest

from multiprocessing import Queue

from openpilot.common.test import OpenpilotTestCase
from openpilot.cereal import messaging
from openpilot.common.basedir import BASEDIR
from openpilot.common.hardware.hw import Paths
from openpilot.tools.sim.bridge.common import QueueMessageType

SIM_DIR = os.path.join(BASEDIR, "openpilot/tools/sim")

class TestSimBridgeBase(OpenpilotTestCase):
  @classmethod
  def setup_class(cls):
    if cls is TestSimBridgeBase:
      raise unittest.SkipTest("Don't run this base class, run test_metadrive_bridge.py instead")

  def setup_method(self):
    self.processes = []

  @unittest.skipUnless(os.environ.get("RUN_METADRIVE_TEST"), "set RUN_METADRIVE_TEST=1 to run the integration test")
  def test_driving(self):
    # Startup manager and bridge.py. Check processes are running, then engage and verify.
    p_manager = subprocess.Popen("./launch_openpilot.sh", cwd=SIM_DIR, start_new_session=True)
    self.manager_process = p_manager
    self.processes.append(p_manager)

    sm = messaging.SubMaster(['selfdriveState', 'onroadEvents', 'managerState', 'modelV2'])
    q = Queue()
    bridge = self.create_bridge()
    p_bridge = bridge.run(q, retries=10)
    self.processes.append(p_bridge)

    max_time_per_step = 60

    # Wait for bridge to startup
    start_waiting = time.monotonic()
    while not bridge.started.value and time.monotonic() < start_waiting + max_time_per_step:
      p_bridge.join(timeout=0)
      if p_bridge.exitcode is not None:
        break
      time.sleep(0.1)
    assert p_bridge.exitcode is None, f"Bridge process should be running, but exited with code {p_bridge.exitcode}"

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

    assert no_car_events_issues_once, \
                    f"Failed because no messages received, or CarEvents '{car_event_issues}' or processes not running '{not_running}'"

    start_time = time.monotonic()
    min_counts_control_active = 100
    control_active = 0

    while time.monotonic() < start_time + max_time_per_step:
      sm.update()

      if sm.all_alive() and sm['selfdriveState'].active:
        control_active += 1

        if control_active == min_counts_control_active:
          break

    assert min_counts_control_active == control_active, f"Simulator did not engage a minimal of {min_counts_control_active} steps was {control_active}"

    failure_states = []
    model_messages = 0
    model_rate_start = time.monotonic()
    while bridge.started.value:
      sm.update(100)
      model_messages += int(sm.updated['modelV2'])

    model_elapsed = time.monotonic() - model_rate_start
    model_rate = model_messages / model_elapsed
    minimum_model_rate = float(os.environ.get("MIN_MODEL_RATE", "18"))
    print(f"modelV2 rate: {model_rate:.2f} Hz over {model_elapsed:.1f} s", flush=True)
    assert model_rate >= minimum_model_rate, f"modelV2 ran at {model_rate:.2f} Hz, below {minimum_model_rate:.2f} Hz"

    while not q.empty():
      state = q.get()
      if state.type == QueueMessageType.TERMINATION_INFO:
        done_info = state.info
        failure_states = [done_state for done_state in done_info if done_state != "timeout" and done_info[done_state]]
        break
    assert len(failure_states) == 0, f"Simulator fails to finish a loop. Failure states: {failure_states}"

  def teardown_method(self):
    print("Test shutting down. CommIssues are acceptable")
    manager_process = getattr(self, "manager_process", None)
    if manager_process is not None and manager_process.poll() is None:
      os.killpg(os.getpgid(manager_process.pid), signal.SIGINT)
      try:
        manager_process.wait(timeout=15)
      except subprocess.TimeoutExpired:
        pass

    if (save_dir := os.environ.get("SIM_LOG_SAVE_DIR")) and Path(Paths.log_root()).is_dir():
      shutil.copytree(Paths.log_root(), save_dir, dirs_exist_ok=True)

    for p in reversed(self.processes):
      if isinstance(p, subprocess.Popen):
        if p.poll() is None:
          p.terminate()
      elif p.is_alive():
        p.terminate()

    for p in reversed(self.processes):
      if isinstance(p, subprocess.Popen):
        try:
          p.wait(timeout=5)
        except subprocess.TimeoutExpired:
          p.kill()
      else:
        p.join(timeout=5)
        if p.is_alive():
          p.kill()
