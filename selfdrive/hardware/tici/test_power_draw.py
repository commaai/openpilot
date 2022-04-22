#!/usr/bin/env python3
import unittest
import numpy as np
import time
import math

from common.realtime import Ratekeeper
from selfdrive.hardware.tici.power_monitor import get_power
from selfdrive.manager.process_config import managed_processes
from selfdrive.manager.manager import manager_cleanup

POWER = {
  'camerad': 2.7,
  'modeld': 1.2,
  'dmonitoringmodeld': 0.25,
  'loggerd': 0.5,
}


class TestPowerDraw(unittest.TestCase):

  def tearDown(self):
    manager_cleanup()

  def test_camera_procs(self):
    baseline = get_power()

    prev = baseline
    used = {}
    for proc in POWER.keys():
      managed_processes[proc].start()
      time.sleep(3)

      now = get_power()
      used[proc] = now - prev
      prev = now

    manager_cleanup()

    print("-"*30)
    print(f"Baseline {baseline}W")
    for proc in POWER.keys():
      cur = used[proc]
      expected = POWER[proc]
      with self.subTest(proc=proc):
        print(f"{proc.ljust(20)} {expected:.2f}W {cur:.2f}W")
        self.assertTrue(math.isclose(cur, expected, rel_tol=0.05, abs_tol=0.1))
    print("-"*30)


if __name__ == "__main__":
  unittest.main()
