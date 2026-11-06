#!/usr/bin/env python3
# end to end fault tests on a DUT in an ultimate provisioning (UP) station, the UP plays the car in car.py.
# setup_class drives once through the faults in faults.py and records each one, the tests check the recordings.
# FAULTS=brake_press,panda_nmi only drives those
import itertools
import json
import math
import os
import re
import signal
import subprocess
import time
import unittest
from dataclasses import dataclass, field

import openpilot.cereal.messaging as messaging
from openpilot.common.basedir import BASEDIR
from openpilot.common.test import OpenpilotTestCase
from openpilot.common.utils import tabulate
from openpilot.selfdrive.test.helpers import set_params_enabled
from openpilot.selfdrive.test.up.car import Car
from openpilot.selfdrive.test.up.faults import FAULTS, PANDA_RESET_FLAGS, Fault

PANDA_TIMEOUT = 5.  # the panda's reaction to openpilot's
PANDA_REPORT_TIMEOUT = 30.  # the self-tests run in the background
SAFETY_MISMATCH_GRACE = 10.  # selfdrived ignores safety mode mismatches for this long after it starts
OBSERVE = 0.5  # openpilot has to hold its state this long after the fault shows up
STAY_ENGAGED = 2.  # a temporary fault is watched at least this long, a late disengagement would be blamed on the next fault
ACTUATION_STOP = 0.2  # carControl's last actuating frame after disengaging
SAMPLE_GAP = 1.  # longer without a sample, a request to the UP or the panda stalled the test


@dataclass
class Run:
  fault: Fault
  t0: float = math.inf  # when the injection returned, a slow request to the UP isn't openpilot's latency
  samples: list[dict] = field(default_factory=list)  # from getting ready for the fault until it's cleared
  log: list[tuple[float, str]] = field(default_factory=list)  # the panda's debug log lines
  error: str = ""  # why the fault couldn't be injected
  stuck: str = ""  # why openpilot didn't come back onroad after it
  unexpected: set[str] = field(default_factory=set)  # panda reports without an injection

  # the samples from the injection until it's cleared
  start: int = 0
  end: int | None = None  # open while recording

  @property
  def window(self) -> list[dict]:
    return self.samples[self.start:self.end] if self.t0 < math.inf else []

  def first(self, cond) -> dict | None:
    return next((s for s in self.window if cond(s)), None)

  @property
  def disengaged(self) -> dict | None:
    # selfdriveState goes stale offroad
    return self.first(lambda s: not s["enabled"] or (self.fault.offroad and not s["started"]))

  @property
  def dropped(self) -> dict | None:
    return self.first(lambda s: not s["controls_allowed"])

  @property
  def reported(self) -> dict | None:
    return self.first(lambda s: self.fault.report in s["panda_reports"])

  @property
  def logged(self) -> bool:
    return all(any(t >= self.t0 and re.search(want, line) for t, line in self.log) for want in self.fault.log)

  @property
  def stopped(self) -> float:
    off, dropped = self.disengaged, self.dropped
    assert off is not None, "never disengaged"
    # offroad comes first, selfdrived disengages on the next pandaStates that has pandad's NO_OUTPUT
    if self.fault.offroad and dropped is not None:
      return max(off["t"], dropped["t"])
    return off["t"]

  def done(self) -> bool:
    expect = self.fault.expect
    reacted = expect is None or (self.disengaged is not None and (not expect.panda_drop or self.dropped is not None))
    return reacted and (not self.fault.report or self.reported is not None) and self.logged

  def timeline(self) -> str:
    rows, last = [], None
    for s in self.samples:
      row = (s["state"], s["events"], s["started"], s["actuating"])
      if s["t"] >= self.t0 - 0.5 and row != last:
        rows.append(f"  {s['t'] - self.t0:+.2f}s {s['state']} {s['events']} started={s['started']} actuating={s['actuating']}")
        last = row
    return "\n".join(rows)


class Drive:
  # openpilot onroad in the UP's car, one fault at a time

  def __init__(self, up):
    self.up = up
    self.car = Car(up)
    self.sm = messaging.SubMaster(['selfdriveState', 'onroadEvents', 'carControl', 'carState', 'pandaStates', 'deviceState', 'managerState'])
    self.log_sock = messaging.sub_sock('logMessage', conflate=False)
    # getting onroad, before the first fault
    self.startup = self.run = Run(Fault("startup", ()))
    self.panda_uptime = 0
    self.injected: set[str] = set()  # the panda reports injected since it booted
    self.rates_broken = False
    self.settled = 0.  # when the CAN is back to normal after an outage
    self.onroad_t = 0.
    self.engaged_t: float | None = None  # engaged since

  def step(self) -> dict:
    self.sm.update(10)
    ps = self.sm['pandaStates'][0] if len(self.sm['pandaStates']) else None
    s = {
      "t": time.monotonic(),
      "started": self.sm['deviceState'].started,
      "state": str(self.sm['selfdriveState'].state),
      "enabled": self.sm['selfdriveState'].enabled,
      "engageable": self.sm['selfdriveState'].engageable,
      "events": [str(e.name) for e in self.sm['onroadEvents']],
      # carControl goes stale when controlsd dies
      "actuating": self.sm.alive['carControl'] and (self.sm['carControl'].latActive or self.sm['carControl'].longActive),
      "can_valid": self.sm['carState'].canValid,
      "pressed": self.sm['carState'].steeringPressed,
      "cruise": self.sm['carState'].cruiseState.enabled,
      # a hung selfdrived or pandad would leave its last state in the samples
      "publishing": all(time.monotonic() - self.sm.recv_time[svc] < 1. for svc in ('selfdriveState', 'pandaStates')),
      "controls_allowed": ps is not None and ps.controlsAllowed,
      # when pandad published it, the sample can be a buffered older message
      "panda_t": self.sm.logMonoTime['pandaStates'] / 1e9,
      "toyota": ps is not None and str(ps.safetyModel) == "toyota",
      "panda_reports": [] if ps is None else [str(f) for f in ps.faults] + [f for f in PANDA_RESET_FLAGS if getattr(ps, f)],
      "all_running": all(p.running for p in self.sm['managerState'].processes if p.shouldBeRunning),
      # the manager stops the onroad processes one by one
      "onroad_stopped": not any(p.running for p in self.sm['managerState'].processes if p.name == "selfdrived" or not p.shouldBeRunning),
    }
    if ps is not None:
      if ps.uptime < self.panda_uptime:
        # only the reset the fault injected carries over the reboot
        self.injected = {self.run.fault.report} if self.run.fault.resets_panda else set()
        self.rates_broken = False
      self.panda_uptime = ps.uptime

    # every panda fault openpilot reports has to come from an injection since the panda booted
    unexpected = set(s["panda_reports"]) - self.injected
    if self.rates_broken:
      unexpected = {r for r in unexpected if not r.startswith("interruptRate")}
    self.injected |= unexpected
    self.run.unexpected |= unexpected

    if not s["enabled"]:
      self.engaged_t = None
    elif self.engaged_t is None:
      self.engaged_t = s["t"]
    self.run.samples.append(s)
    for m in messaging.drain_sock(self.log_sock):
      msg = json.loads(m.logMessage).get("msg")
      if isinstance(msg, str):
        self.run.log += [(s["t"], line.strip()) for line in msg.splitlines()]
    return s

  @property
  def last(self) -> dict:
    return self.run.samples[-1]

  def wait_for(self, cond, timeout: float, what: str) -> dict:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
      if cond(s := self.step()):
        return s
    raise TimeoutError(f"no {what} after {timeout:.1f}s, last: {self.last}")

  def hold(self, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
      self.step()

  def signal(self, name: str, sig: signal.Signals) -> None:
    pid = next(p.pid for p in self.sm['managerState'].processes if p.name == name)
    # 0 is our own process group
    assert pid > 0, f"{name} not running"
    os.kill(pid, sig)

  def wait_onroad(self) -> None:
    s = self.wait_for(lambda s: s["started"] and s["can_valid"] and s["toyota"] and s["all_running"], 120, "onroad")
    self.onroad_t = s["t"]

  def ignition_cycle(self) -> None:
    self.up.set_ignition(False)
    # until the manager stopped the onroad processes, a quick ignition on would keep them running
    self.wait_for(lambda s: not s["started"] and not s["toyota"] and s["onroad_stopped"], 15, "offroad")
    self.up.set_ignition(True)
    self.wait_onroad()

  def engage(self) -> None:
    if self.last["enabled"]:
      return
    for _ in range(10):
      # openpilot engages on cruise's rising edge
      self.car.set(cruise=False)
      self.wait_for(lambda s: not s["cruise"] and s["engageable"], 60, "engageable")
      self.car.set(cruise=True)
      try:
        self.wait_for(lambda s: s["state"] == "enabled", 4, "engagement")
        return
      except TimeoutError:
        pass
    raise TimeoutError(f"couldn't engage, last: {self.last}")

  def get_ready(self, fault: Fault) -> None:
    self.car.set(touch=fault.touch)
    # the replay's touch pulses stop once the call returns, the samples can miss one that was in flight
    touch_off = time.monotonic()
    self.hold(self.settled - time.monotonic())
    self.engage()
    # the panda resets to SILENT, selfdrived only sees the safety mode mismatch after its grace
    if fault.resets_panda:
      self.hold(self.onroad_t + SAFETY_MISMATCH_GRACE - time.monotonic())

    # still engaged from the last fault counts
    assert self.engaged_t is not None
    ready = self.engaged_t + 0.5
    if not fault.touch:
      # selfdrived counts the wheel as touched for 2 s after it's released, and from when it initialized
      ready = max(ready, touch_off + 2.1, self.onroad_t + 2.1)
    self.hold(ready - time.monotonic())

    before = [s for s in self.run.samples if s["t"] >= ready - 0.5]
    assert all(s["enabled"] for s in before), f"not engaged before the fault: {self.last}"
    assert fault.touch or not any(s["pressed"] for s in before), "wheel touched before the fault"
    assert fault.report not in self.last["panda_reports"], f"panda {fault.report} before the fault"

  def record(self, fault: Fault) -> Run:
    run = self.run = Run(fault)
    self.step()
    try:
      self.get_ready(fault)
      if fault.report:
        self.injected.add(fault.report)
      self.injected.update(fault.also_reports)
      self.rates_broken |= fault.breaks_rates
      self.step()
      for i in fault.inject:
        i.inject(self)
      run.t0, run.start = time.monotonic(), len(run.samples)
      end = run.t0 + PANDA_REPORT_TIMEOUT
      while not run.done() and time.monotonic() < end:
        self.step()
      self.hold(OBSERVE)
      if fault.expect is None:
        self.hold(run.t0 + STAY_ENGAGED - time.monotonic())
      # openpilot can't be judged on a recording with a hole in it, from right before the injection on
      gap = max(b["t"] - a["t"] for a, b in itertools.pairwise(run.samples[run.start - 1:]))
      assert gap < SAMPLE_GAP, f"no samples for {gap:.1f}s, the test stalled"
    except Exception as e:
      run.error = f"{type(e).__name__}: {e}"
    finally:
      run.end = len(run.samples)
      # cruise off first, clearing pcm_cancel with cruise still on is a rising edge that engages
      if fault.expect is not None:
        self.car.set(cruise=False)
      for i in fault.inject:
        i.clear(self)
    return run

  def record_all(self, faults: list[Fault]) -> dict[str, Run]:
    self.up.set_ignition(True)
    self.wait_onroad()
    runs = {}
    for fault in faults:
      run = runs[fault.name] = self.record(fault)
      # a failure can leave openpilot broken, the next fault starts from a fresh onroad
      if fault.latched or run.error or not run.done():
        try:
          self.ignition_cycle()
        except TimeoutError as e:
          run.stuck = str(e)
          break
    return runs


class TestUPFaults(OpenpilotTestCase):
  COMMA_HARDWARE_TEST = True

  def setUp(self):
    # the hardware is set up once for the whole drive in setup_class
    unittest.TestCase.setUp(self)

  @classmethod
  def setup_class(cls):
    from ultimate_provisioning import UP  # only on a DUT, from its UP's up run or up ssh
    names = os.getenv("FAULTS")
    cls.faults = [f for f in FAULTS if names is None or f.name in names.split(",")]

    cls.drive = drive = Drive(UP.remote())
    set_params_enabled()
    os.environ['SKIP_FW_QUERY'] = '1'
    os.environ['TESTING_CLOSET'] = '1'
    os.environ['BLOCK'] = 'soundd'
    manager = subprocess.Popen(["python", os.path.join(BASEDIR, "openpilot/system/manager/manager.py")])
    try:
      cls.runs = drive.record_all(cls.faults)
    finally:
      manager.send_signal(signal.SIGINT)
      try:
        manager.wait(60)
      except subprocess.TimeoutExpired:
        manager.kill()
        manager.wait()

    rows = []
    for run in cls.runs.values():
      disengaged = "" if run.disengaged is None else f"{run.disengaged['t'] - run.t0:.2f}"
      reported = "" if run.reported is None else f"{run.reported['t'] - run.t0:.2f}"
      took = run.samples[-1]["t"] - run.samples[0]["t"] if run.samples else 0.
      rows.append((run.fault.name, disengaged, reported, f"{took:.1f}", run.error[:60]))
    print(tabulate(rows, ("fault", "disengaged (s)", "reported (s)", "took (s)", "error"), tablefmt="simple_grid"))

  def recorded(self, cond=lambda f: True) -> list[Run]:
    # the faults that got injected, the others fail test_injected
    return [r for r in self.runs.values() if not r.error and cond(r.fault)]

  def test_injected(self, subtests):
    for fault in self.faults:
      with subtests.test(fault=fault.name):
        assert fault.name in self.runs, "not recorded, an earlier fault left openpilot offroad"
        assert not self.runs[fault.name].error, self.runs[fault.name].error

  def test_back_onroad(self, subtests):
    for run in self.runs.values():
      with subtests.test(fault=run.fault.name):
        assert not run.stuck, f"openpilot didn't come back onroad after an ignition cycle: {run.stuck}"

  def test_stays_engaged(self, subtests):
    for run in self.recorded(lambda f: f.expect is None):
      with subtests.test(fault=run.fault.name):
        assert all(s["enabled"] for s in run.window), f"disengaged on a temporary fault\n{run.timeline()}"
        assert all(s["publishing"] for s in run.window), "selfdrived or pandad stopped publishing"

  def test_disengages(self, subtests):
    for run in self.recorded(lambda f: f.expect is not None):
      with subtests.test(fault=run.fault.name):
        assert run.disengaged is not None, f"still engaged\n{run.timeline()}"
        latency = run.disengaged["t"] - run.t0
        assert latency <= run.fault.expect.within, f"disengaged after {latency:.2f}s > {run.fault.expect.within}s\n{run.timeline()}"

  def test_events(self, subtests):
    for run in self.recorded(lambda f: f.expect is not None and f.expect.events):
      with subtests.test(fault=run.fault.name):
        events = {e for s in run.window for e in s["events"]}
        assert events & run.fault.expect.events, f"expected one of {set(run.fault.expect.events)}\n{run.timeline()}"

  def test_soft_disable(self, subtests):
    for run in self.recorded(lambda f: f.expect is not None and f.expect.soft):
      with subtests.test(fault=run.fault.name):
        assert any(s["state"] == "softDisabling" for s in run.window), f"expected softDisabling\n{run.timeline()}"

  def test_stops_actuating(self, subtests):
    for run in self.recorded(lambda f: f.expect is not None):
      with subtests.test(fault=run.fault.name):
        assert run.disengaged is not None, "never disengaged"
        last = max((s["t"] for s in run.window if s["actuating"]), default=0.)
        assert last <= run.stopped + ACTUATION_STOP, f"still actuating {last - run.stopped:.2f}s after disengaging\n{run.timeline()}"

  def test_panda_drops_controls(self, subtests):
    for run in self.recorded(lambda f: f.expect is not None and f.expect.panda_drop):
      with subtests.test(fault=run.fault.name):
        assert run.disengaged is not None, "never disengaged"
        assert run.dropped is not None, f"panda kept controls_allowed\n{run.timeline()}"
        assert run.dropped["t"] - run.disengaged["t"] <= PANDA_TIMEOUT, "panda dropped controls_allowed late"

  def test_panda_keeps_controls(self, subtests):
    for run in self.recorded(lambda f: f.expect is None):
      with subtests.test(fault=run.fault.name):
        panda = [s for s in run.window if s["panda_t"] > run.t0]
        assert panda and all(s["controls_allowed"] for s in panda), "panda dropped controls_allowed on a temporary fault"

  def test_panda_reports(self, subtests):
    for run in self.recorded(lambda f: f.report):
      with subtests.test(fault=run.fault.name):
        assert run.reported is not None, f"no panda {run.fault.report}\n{run.timeline()}"
        if run.fault.expect is not None:
          timeout = run.fault.expect.within + PANDA_TIMEOUT
          assert run.reported["t"] - run.t0 <= timeout, f"panda {run.fault.report} after {run.reported['t'] - run.t0:.1f}s"

  def test_panda_logs(self, subtests):
    for run in self.recorded(lambda f: f.log):
      with subtests.test(fault=run.fault.name):
        assert run.logged, f"no panda log {run.fault.log}, got: {[line for _, line in run.log]}"

  def test_no_unexpected_panda_reports(self, subtests):
    for run in [self.drive.startup, *self.runs.values()]:
      with subtests.test(fault=run.fault.name):
        assert not run.unexpected, f"panda reported {sorted(run.unexpected)} without an injection"


if __name__ == "__main__":
  unittest.main()
