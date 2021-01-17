#!/usr/bin/env python3
import os
import time
import subprocess
import unittest
from pathlib import Path

import cereal.messaging as messaging
from common.basedir import BASEDIR
from common.timeout import Timeout
from selfdrive.loggerd.config import ROOT
from selfdrive.test.helpers import set_params_enabled
from tools.lib.logreader import LogReader

PROCS = [
  ("selfdrive.controls.controlsd", 47.0),
  ("./loggerd", 45.0),
  ("selfdrive.locationd.locationd", 35.0),
  ("selfdrive.controls.plannerd", 20.0),
  ("selfdrive.locationd.paramsd", 12.0),
  ("./_modeld", 7.12),
  ("./camerad", 7.07),
  ("./_sensord", 6.17),
  ("./_ui", 5.82),
  ("selfdrive.controls.radard", 5.67),
  ("./boardd", 3.63),
  ("./_dmonitoringmodeld", 2.67),
  ("selfdrive.logmessaged", 1.7),
  ("selfdrive.thermald.thermald", 2.41),
  ("selfdrive.locationd.calibrationd", 2.0),
  ("selfdrive.monitoring.dmonitoringd", 1.90),
  ("./proclogd", 1.54),
  ("./_gpsd", 0.09),
  ("./clocksd", 0.02),
  ("./ubloxd", 0.02),
  ("selfdrive.tombstoned", 0),
  ("./logcatd", 0),
]

# ***** test helpers *****

def cputime_total(ct):
  return ct.cpuUser + ct.cpuSystem + ct.cpuChildrenUser + ct.cpuChildrenSystem

def check_cpu_usage(first_proc, last_proc):
  result =  "------------------------------------------------\n"
  result += "------------------ CPU Usage -------------------\n"
  result += "------------------------------------------------\n"

  r = True
  dt = (last_proc.logMonoTime - first_proc.logMonoTime) / 1e9
  for proc_name, normal_cpu_usage in PROCS:
    first, last = None, None
    try:
      first = [p for p in first_proc.procLog.procs if proc_name in p.cmdline][0]
      last = [p for p in last_proc.procLog.procs if proc_name in p.cmdline][0]
      cpu_time = cputime_total(last) - cputime_total(first)
      cpu_usage = cpu_time / dt * 100.
      if cpu_usage > max(normal_cpu_usage * 1.1, normal_cpu_usage + 5.0):
        result += f"Warning {proc_name} using more CPU than normal\n"
        r = False
      elif cpu_usage < min(normal_cpu_usage * 0.65, max(normal_cpu_usage - 1.0, 0.0)):
        result += f"Warning {proc_name} using less CPU than normal\n"
        r = False
      result += f"{proc_name.ljust(35)}  {cpu_usage:.2f}%\n"
    except IndexError:
      result += f"{proc_name.ljust(35)}  NO METRICS FOUND {first=} {last=}\n"
      r = False
  result += "------------------------------------------------\n"
  print(result)
  return r


class TestOnroad(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    os.environ['SKIP_FW_QUERY'] = "1"
    os.environ['FINGERPRINT'] = "TOYOTA COROLLA TSS2 2019"
    set_params_enabled()

    initial_segments = set(Path(ROOT).iterdir())

    # start manager and run openpilot for a minute
    try:
      manager_path = os.path.join(BASEDIR, "selfdrive/manager.py")
      proc = subprocess.Popen(["python", manager_path])

      sm = messaging.SubMaster(['carState'])
      with Timeout(60, "controls didn't start"):
        while not sm.updated['carState']:
          sm.update(1000)

      time.sleep(60)
    finally:
      proc.terminate()
      if proc.wait(20) is None:
        proc.kill()

    new_segments = set(Path(ROOT).iterdir()) - initial_segments

    segments = [p for p in new_segments if len(list(p.iterdir())) > 1]
    cls.segment = [s for s in segments if str(s).endswith("--0")][0]
    cls.lr = list(LogReader(os.path.join(str(cls.segment), "rlog.bz2")))

  def test_cpu_usage(self):
    proclogs = [m for m in self.lr if m.which() == 'procLog']
    cpu_ok = check_cpu_usage(proclogs[5], proclogs[-3])
    self.assertTrue(cpu_ok)

if __name__ == "__main__":
  unittest.main()
