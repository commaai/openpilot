#!/usr/bin/env python3
import unittest
import time
import math
import threading
from dataclasses import dataclass
from tabulate import tabulate

import cereal.messaging as messaging
from system.hardware import HARDWARE, TICI
from system.hardware.tici.power_monitor import get_power
from selfdrive.manager.process_config import managed_processes
from selfdrive.manager.manager import manager_cleanup


@dataclass
class Proc:
  name: str
  power: float
  rtol: float = 0.05
  atol: float = 0.12
  warmup: float = 6.

PROCS = [
  Proc('camerad', 2.1),
  Proc('modeld', 0.93, atol=0.2),
  Proc('dmonitoringmodeld', 0.4),
  Proc('encoderd', 0.23),
  Proc('mapsd', 0.05),
  Proc('navmodeld', 0.05),
]

def send_llk_msg(done):
  pm = messaging.PubMaster(['liveLocationKalman'])
  msg = messaging.new_message('liveLocationKalman')
  msg.liveLocationKalman.positionGeodetic = {'value': [32.7174, -117.16277, 0], 'std': [0., 0., 0.], 'valid': True}
  msg.liveLocationKalman.calibratedOrientationNED = {'value': [0., 0., 0.], 'std': [0., 0., 0.], 'valid': True}
  msg.liveLocationKalman.status = 'valid'

  # Send liveLocationKalman at 20hz
  while not done.is_set():
    pm.send('liveLocationKalman', msg)
    time.sleep(1/20)


class TestPowerDraw(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    if not TICI:
      raise unittest.SkipTest

  def setUp(self):
    HARDWARE.initialize_hardware()
    HARDWARE.set_power_save(False)

    # wait a bit for power save to disable
    time.sleep(5)

  def tearDown(self):
    manager_cleanup()

  def test_camera_procs(self):
    baseline = get_power()
    done = threading.Event()
    thread = threading.Thread(target=send_llk_msg, args=(done,), daemon=True)
    thread.start()

    prev = baseline
    used = {}
    for proc in PROCS:
      managed_processes[proc.name].start()
      time.sleep(proc.warmup)

      now = get_power(8)
      used[proc.name] = now - prev
      prev = now

    done.set()
    manager_cleanup()

    tab = []
    tab.append(['process', 'expected (W)', 'measured (W)'])
    for proc in PROCS:
      cur = used[proc.name]
      expected = proc.power
      tab.append([proc.name, round(expected, 2), round(cur, 2)])
      with self.subTest(proc=proc.name):
        self.assertTrue(math.isclose(cur, expected, rel_tol=proc.rtol, abs_tol=proc.atol))
    print(tabulate(tab))
    print(f"Baseline {baseline:.2f}W\n")


if __name__ == "__main__":
  unittest.main()
