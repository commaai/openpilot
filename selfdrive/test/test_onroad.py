#!/usr/bin/env python3
import json
import os
import subprocess
import time
import numpy as np
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from cereal import car
import cereal.messaging as messaging
from cereal.services import service_list
from common.basedir import BASEDIR
from common.timeout import Timeout
from common.params import Params
from selfdrive.controls.lib.events import EVENTS, ET
from system.loggerd.config import ROOT
from selfdrive.test.helpers import set_params_enabled, release_only
from tools.lib.logreader import LogReader

# Baseline CPU usage by process
PROCS = {
  "selfdrive.controls.controlsd": 39.0,
  "./loggerd": 10.0,
  "./encoderd": 17.0,
  "./camerad": 14.5,
  "./locationd": 9.1,
  "selfdrive.controls.plannerd": 16.5,
  "./_ui": 19.2,
  "selfdrive.locationd.paramsd": 9.0,
  "./_sensord": 12.0,
  "selfdrive.controls.radard": 4.5,
  "./_modeld": 4.48,
  "./boardd": 3.63,
  "./_dmonitoringmodeld": 5.0,
  "selfdrive.thermald.thermald": 3.87,
  "selfdrive.locationd.calibrationd": 2.0,
  "selfdrive.locationd.torqued": 5.0,
  "./_soundd": 1.0,
  "selfdrive.monitoring.dmonitoringd": 4.0,
  "./proclogd": 1.54,
  "system.logmessaged": 0.2,
  "./clocksd": 0.02,
  "./ubloxd": 0.02,
  "selfdrive.tombstoned": 0,
  "./logcatd": 0,
}

TIMINGS = {
  # rtols: max/min, rsd
  "can": [2.5, 0.35],
  "pandaStates": [2.5, 0.35],
  "peripheralState": [2.5, 0.35],
  "sendcan": [2.5, 0.35],
  "carState": [2.5, 0.35],
  "carControl": [2.5, 0.35],
  "controlsState": [2.5, 0.35],
  "lateralPlan": [2.5, 0.5],
  "longitudinalPlan": [2.5, 0.5],
  "roadCameraState": [2.5, 0.35],
  "driverCameraState": [2.5, 0.35],
  "modelV2": [2.5, 0.35],
  "driverStateV2": [2.5, 0.40],
  "liveLocationKalman": [2.5, 0.35],
  "wideRoadCameraState": [1.5, 0.35],
}


def cputime_total(ct):
  return ct.cpuUser + ct.cpuSystem + ct.cpuChildrenUser + ct.cpuChildrenSystem


def check_cpu_usage(proclogs):
  result = "\n"
  result += "------------------------------------------------\n"
  result += "------------------ CPU Usage -------------------\n"
  result += "------------------------------------------------\n"

  plogs_by_proc = defaultdict(list)
  for pl in proclogs:
    for x in pl.procLog.procs:
      if len(x.cmdline) > 0:
        n = list(x.cmdline)[0]
        plogs_by_proc[n].append(x)

  print(plogs_by_proc.keys())

  r = True
  dt = (proclogs[-1].logMonoTime - proclogs[0].logMonoTime) / 1e9
  for proc_name, expected_cpu in PROCS.items():
    err = ""
    cpu_usage = 0.
    x = plogs_by_proc[proc_name]
    if len(x) > 2:
      cpu_time = cputime_total(x[-1]) - cputime_total(x[0])
      cpu_usage = cpu_time / dt * 100.
      if cpu_usage > max(expected_cpu * 1.15, expected_cpu + 5.0):
        # cpu usage is high while playing sounds
        if not (proc_name == "./_soundd" and cpu_usage < 65.):
          err = "using more CPU than normal"
      elif cpu_usage < min(expected_cpu * 0.65, max(expected_cpu - 1.0, 0.0)):
        err = "using less CPU than normal"
    else:
      err = "NO METRICS FOUND"

    result += f"{proc_name.ljust(35)}  {cpu_usage:5.2f}% ({expected_cpu:5.2f}%) {err}\n"
    if len(err) > 0:
      r = False

  result += "------------------------------------------------\n"
  print(result)
  return r


class TestOnroad(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    if "DEBUG" in os.environ:
      segs = filter(lambda x: os.path.exists(os.path.join(x, "rlog")), Path(ROOT).iterdir())
      segs = sorted(segs, key=lambda x: x.stat().st_mtime)
      print(segs[-2])
      cls.lr = list(LogReader(os.path.join(segs[-2], "rlog")))
      return

    # setup env
    os.environ['REPLAY'] = "1"
    os.environ['SKIP_FW_QUERY'] = "1"
    os.environ['FINGERPRINT'] = "TOYOTA COROLLA TSS2 2019"
    os.environ['LOGPRINT'] = 'debug'

    params = Params()
    params.clear_all()
    set_params_enabled()

    # Make sure athena isn't running
    os.system("pkill -9 -f athena")

    # start manager and run openpilot for a minute
    proc = None
    try:
      manager_path = os.path.join(BASEDIR, "selfdrive/manager/manager.py")
      proc = subprocess.Popen(["python", manager_path])

      sm = messaging.SubMaster(['carState'])
      with Timeout(150, "controls didn't start"):
        while sm.rcv_frame['carState'] < 0:
          sm.update(1000)

      # make sure we get at least two full segments
      route = None
      cls.segments = []
      with Timeout(300, "timed out waiting for logs"):
        while route is None:
          route = params.get("CurrentRoute", encoding="utf-8")
          time.sleep(0.1)

        while len(cls.segments) < 3:
          segs = set()
          if Path(ROOT).exists():
            segs = set(Path(ROOT).glob(f"{route}--*"))
          cls.segments = sorted(segs, key=lambda s: int(str(s).rsplit('--')[-1]))
          time.sleep(2)

      # chop off last, incomplete segment
      cls.segments = cls.segments[:-1]

    finally:
      if proc is not None:
        proc.terminate()
        if proc.wait(60) is None:
          proc.kill()

    cls.lrs = [list(LogReader(os.path.join(str(s), "rlog"))) for s in cls.segments]

    # use the second segment by default as it's the first full segment
    cls.lr = list(LogReader(os.path.join(str(cls.segments[1]), "rlog")))

  def test_cloudlog_size(self):
    msgs = [m for m in self.lr if m.which() == 'logMessage']

    total_size = sum(len(m.as_builder().to_bytes()) for m in msgs)
    self.assertLess(total_size, 3.5e5)

    cnt = Counter(json.loads(m.logMessage)['filename'] for m in msgs)
    big_logs = [f for f, n in cnt.most_common(3) if n / sum(cnt.values()) > 30.]
    self.assertEqual(len(big_logs), 0, f"Log spam: {big_logs}")

  def test_ui_timings(self):
    result = "\n"
    result += "------------------------------------------------\n"
    result += "-------------- UI Draw Timing ------------------\n"
    result += "------------------------------------------------\n"

    ts = [m.uiDebug.drawTimeMillis for m in self.lr if m.which() == 'uiDebug']
    result += f"min  {min(ts):.2f}ms\n"
    result += f"max  {max(ts):.2f}ms\n"
    result += f"std  {np.std(ts):.2f}ms\n"
    result += f"mean {np.mean(ts):.2f}ms\n"
    result += "------------------------------------------------\n"
    print(result)

    self.assertGreater(len(ts), 20*50, "insufficient samples")
    #self.assertLess(max(ts), 30.)
    self.assertLess(np.mean(ts), 10.)
    #self.assertLess(np.std(ts), 5.)

  def test_cpu_usage(self):
    proclogs = [m for m in self.lr if m.which() == 'procLog']
    self.assertGreater(len(proclogs), service_list['procLog'].frequency * 45, "insufficient samples")
    cpu_ok = check_cpu_usage(proclogs)
    self.assertTrue(cpu_ok)

  def test_camera_processing_time(self):
    result = "\n"
    result += "------------------------------------------------\n"
    result += "-------------- Debayer Timing ------------------\n"
    result += "------------------------------------------------\n"

    ts = [getattr(getattr(m, m.which()), "processingTime") for m in self.lr if 'CameraState' in m.which()]
    self.assertLess(min(ts), 0.025, f"high execution time: {min(ts)}")
    result += f"execution time: min  {min(ts):.5f}s\n"
    result += f"execution time: max  {max(ts):.5f}s\n"
    result += f"execution time: mean {np.mean(ts):.5f}s\n"
    result += "------------------------------------------------\n"
    print(result)

  def test_mpc_execution_timings(self):
    result = "\n"
    result += "------------------------------------------------\n"
    result += "-----------------  MPC Timing ------------------\n"
    result += "------------------------------------------------\n"

    cfgs = [("lateralPlan", 0.05, 0.05), ("longitudinalPlan", 0.05, 0.05)]
    for (s, instant_max, avg_max) in cfgs:
      ts = [getattr(getattr(m, s), "solverExecutionTime") for m in self.lr if m.which() == s]
      self.assertLess(max(ts), instant_max, f"high '{s}' execution time: {max(ts)}")
      self.assertLess(np.mean(ts), avg_max, f"high avg '{s}' execution time: {np.mean(ts)}")
      result += f"'{s}' execution time: min  {min(ts):.5f}s\n"
      result += f"'{s}' execution time: max  {max(ts):.5f}s\n"
      result += f"'{s}' execution time: mean {np.mean(ts):.5f}s\n"
    result += "------------------------------------------------\n"
    print(result)

  def test_model_execution_timings(self):
    result = "\n"
    result += "------------------------------------------------\n"
    result += "----------------- Model Timing -----------------\n"
    result += "------------------------------------------------\n"
    # TODO: this went up when plannerd cpu usage increased, why?
    cfgs = [
      ("modelV2", 0.050, 0.036),
      ("driverStateV2", 0.050, 0.026),
    ]
    for (s, instant_max, avg_max) in cfgs:
      ts = [getattr(getattr(m, s), "modelExecutionTime") for m in self.lr if m.which() == s]
      self.assertLess(max(ts), instant_max, f"high '{s}' execution time: {max(ts)}")
      self.assertLess(np.mean(ts), avg_max, f"high avg '{s}' execution time: {np.mean(ts)}")
      result += f"'{s}' execution time: min  {min(ts):.5f}s\n"
      result += f"'{s}' execution time: max {max(ts):.5f}s\n"
      result += f"'{s}' execution time: mean {np.mean(ts):.5f}s\n"
    result += "------------------------------------------------\n"
    print(result)

  def test_timings(self):
    passed = True
    result = "\n"
    result += "------------------------------------------------\n"
    result += "----------------- Service Timings --------------\n"
    result += "------------------------------------------------\n"
    for s, (maxmin, rsd) in TIMINGS.items():
      msgs = [m.logMonoTime for m in self.lr if m.which() == s]
      if not len(msgs):
        raise Exception(f"missing {s}")

      ts = np.diff(msgs) / 1e9
      dt = 1 / service_list[s].frequency

      try:
        np.testing.assert_allclose(np.mean(ts), dt, rtol=0.03, err_msg=f"{s} - failed mean timing check")
        np.testing.assert_allclose([np.max(ts), np.min(ts)], dt, rtol=maxmin, err_msg=f"{s} - failed max/min timing check")
      except Exception as e:
        result += str(e) + "\n"
        passed = False

      if np.std(ts) / dt > rsd:
        result += f"{s} - failed RSD timing check\n"
        passed = False

      result += f"{s.ljust(40)}: {np.array([np.mean(ts), np.max(ts), np.min(ts)])*1e3}\n"
      result += f"{''.ljust(40)}  {np.max(np.absolute([np.max(ts)/dt, np.min(ts)/dt]))} {np.std(ts)/dt}\n"
    result += "="*67
    print(result)
    self.assertTrue(passed)

  @release_only
  def test_startup(self):
    startup_alert = None
    for msg in self.lrs[0]:
      # can't use carEvents because the first msg can be dropped while loggerd is starting up
      if msg.which() == "controlsState":
        startup_alert = msg.controlsState.alertText1
        break
    expected = EVENTS[car.CarEvent.EventName.startup][ET.PERMANENT].alert_text_1
    self.assertEqual(startup_alert, expected, "wrong startup alert")


if __name__ == "__main__":
  unittest.main()
