#!/usr/bin/env python3
import os
import math
import shutil
import subprocess
import time
import unittest
from dataclasses import dataclass

from common.basedir import BASEDIR
from common.timeout import Timeout
from common.params import Params
from selfdrive.hardware import HARDWARE, TICI
from selfdrive.hardware.tici.power_monitor import get_power, sample_power
from selfdrive.loggerd.config import ROOT
from selfdrive.manager.process_config import managed_processes
from selfdrive.manager.manager import manager_cleanup


@dataclass
class Proc:
  name: str
  power: float
  rtol: float = 0.05
  atol: float = 0.1
  warmup: float = 3.

PROCS = [
  Proc('camerad', 2.5),
  Proc('modeld', 0.95),
  Proc('dmonitoringmodeld', 0.25),
  Proc('loggerd', 0.45, warmup=10.),
]


class TestPowerDraw(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    if not TICI:
      raise unittest.SkipTest

  def setUp(self):
    HARDWARE.initialize_hardware()
    HARDWARE.set_power_save(False)

    if os.path.isdir(ROOT):
      shutil.rmtree(ROOT)

  def tearDown(self):
    manager_cleanup()

  def test_offroad_power(self):
    proc = None
    try:
      # force offroad
      Params().delete("HasAcceptedTerms")

      manager_path = os.path.join(BASEDIR, "selfdrive/manager/manager.py")
      proc = subprocess.Popen(["python", manager_path])
      time.sleep(45)
      for n in range(15):
        print("measuring", n)
        import numpy as np
        pwrs = sample_power(10)
        print("  ", np.mean(pwrs), np.std(pwrs), np.min(pwrs), np.max(pwrs))
        time.sleep(3)
    finally:
      if proc is not None:
        proc.terminate()
        if proc.wait(20) is None:
          proc.kill()

  def test_camera_procs(self):
    baseline = get_power()

    prev = baseline
    used = {}
    for proc in PROCS:
      managed_processes[proc.name].start()
      time.sleep(proc.warmup)

      now = get_power(8)
      used[proc.name] = now - prev
      prev = now

    manager_cleanup()

    print("-"*35)
    print(f"Baseline {baseline:.2f}W\n")
    for proc in PROCS:
      cur = used[proc.name]
      expected = proc.power
      print(f"{proc.name.ljust(20)} {expected:.2f}W  {cur:.2f}W")
      with self.subTest(proc=proc.name):
        self.assertTrue(math.isclose(cur, expected, rel_tol=proc.rtol, abs_tol=proc.atol))
    print("-"*35)


if __name__ == "__main__":
  unittest.main()
