#!/usr/bin/env python3
import unittest
import time
import math
from collections import OrderedDict

from selfdrive.hardware import HARDWARE, TICI
from selfdrive.hardware.tici.power_monitor import get_power
from selfdrive.manager.process_config import managed_processes
from selfdrive.manager.manager import manager_cleanup

POWER = OrderedDict(
  camerad=2.58,
  modeld=0.90,
  dmonitoringmodeld=0.25,
  loggerd=0.45,
)


class TestPowerDraw(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    if not TICI:
      raise unittest.SkipTest

  def setUp(self):
    HARDWARE.initialize_hardware()
    HARDWARE.set_power_save(False)

  def tearDown(self):
    manager_cleanup()

  def test_camera_procs(self):
    baseline = get_power()

    prev = baseline
    used = {}
    for proc in POWER.keys():
      managed_processes[proc].start()
      time.sleep(6)

      now = get_power(8)
      used[proc] = now - prev
      prev = now

    manager_cleanup()

    print("-"*35)
    print(f"Baseline {baseline:.2f}W\n")
    for proc in POWER.keys():
      cur = used[proc]
      expected = POWER[proc]
      print(f"{proc.ljust(20)} {expected:.2f}W  {cur:.2f}W")
      with self.subTest(proc=proc):
        self.assertTrue(math.isclose(cur, expected, rel_tol=0.10, abs_tol=0.1))
    print("-"*35)


if __name__ == "__main__":
  unittest.main()
