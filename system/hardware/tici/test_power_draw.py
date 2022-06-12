#!/usr/bin/env python3
import unittest
import time
import math
from dataclasses import dataclass

from system.hardware import HARDWARE, TICI
from system.hardware.tici.power_monitor import get_power
from selfdrive.manager.process_config import managed_processes
from selfdrive.manager.manager import manager_cleanup


@dataclass
class Proc:
  name: str
  power: float
  rtol: float = 0.05
  atol: float = 0.1
  warmup: float = 6.

PROCS = [
  Proc('camerad', 2.15),
  Proc('modeld', 1.0),
  Proc('dmonitoringmodeld', 0.25),
  Proc('encoderd', 0.23),
]


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
