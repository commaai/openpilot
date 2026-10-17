import queue
import time
import unittest
from multiprocessing import Queue

from openpilot.cereal import messaging
from openpilot.common.test import OpenpilotTestCase
from openpilot.tools.sim.run import launch_openpilot, stop_openpilot
from openpilot.tools.sim.simulator import start_simulator

ROUTE = "8494c69d3c710e81|000001d4--2648a9a404/5"  # highway, a gentle curve and a lead throughout
START_SEC = 15  # the logged driver tailgates before this, and openpilot can't open up the gap
ENGAGE_TIMEOUT = 40
TIMEOUT = 110
MAX_LATERAL_ERROR = 0.75


class TestSim(OpenpilotTestCase):
  SHARED_DOWNLOAD_CACHE = True

  def setup_method(self):
    self.manager = launch_openpilot(None)
    self.simulator = None

  def teardown_method(self):
    if self.simulator is not None:
      self.simulator.terminate()
      self.simulator.join()
    stop_openpilot(self.manager)

  def test_driving(self):
    sm = messaging.SubMaster(['selfdriveState', 'onroadEvents', 'managerState'])
    # the launch script resets CalibrationParams, the simulator must write the route's after it
    while not sm.seen['managerState']:
      sm.update(100)

    q: Queue = Queue()
    self.simulator = start_simulator(ROUTE, START_SEC, Queue(), q)

    start = time.monotonic()
    engaged_at = None
    while True:
      assert time.monotonic() - start < TIMEOUT, "timed out"
      alive = self.simulator.is_alive()
      try:
        kind, info = q.get(timeout=0.1 if alive else 1)
        assert kind != "error", info
        if kind == "episode_end":
          done = info
          break
      except queue.Empty:
        assert alive, f"simulator exited with {self.simulator.exitcode} before finishing"

      sm.update(0)
      if engaged_at is None:
        if sm['selfdriveState'].active:
          engaged_at = time.monotonic()
        assert time.monotonic() - start < ENGAGE_TIMEOUT, "openpilot did not engage"
        continue

      not_running = [p.name for p in sm['managerState'].processes if p.shouldBeRunning and not p.running]
      assert not not_running, f"processes not running: {not_running}"
      events = [e.name for e in sm['onroadEvents'] if e.noEntry or e.softDisable or e.immediateDisable]
      assert not events, f"disengagement events: {events}"
      assert sm['selfdriveState'].active, f"disengaged {time.monotonic() - engaged_at:.1f}s after engaging"

    assert done["ended"] == "route_end", f"episode ended early: {done}"
    assert done["engaged_time"] > 30, done
    assert done["max_lateral_error"] < MAX_LATERAL_ERROR, done
    assert done["min_lead_gap"] > 0, done


if __name__ == "__main__":
  unittest.main()
