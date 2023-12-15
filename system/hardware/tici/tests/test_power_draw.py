#!/usr/bin/env python3
import pytest
import unittest
import time
import threading
import numpy as np
from dataclasses import dataclass
from tabulate import tabulate
from typing import List

import cereal.messaging as messaging
from cereal.services import SERVICE_LIST
from openpilot.system.hardware import HARDWARE
from openpilot.system.hardware.tici.power_monitor import get_power
from openpilot.selfdrive.manager.process_config import managed_processes
from openpilot.selfdrive.manager.manager import manager_cleanup
from openpilot.selfdrive.navd.tests.test_map_renderer import gen_llk

SAMPLE_TIME = 8   # seconds to sample power

@dataclass
class Proc:
  name: str
  power: float
  msgs: List[str]
  rtol: float = 0.05
  atol: float = 0.12
  warmup: float = 6.

PROCS = [
  Proc('camerad', 2.1, msgs=['roadCameraState', 'wideRoadCameraState', 'driverCameraState']),
  Proc('modeld', 1.12, atol=0.2, msgs=['modelV2']),
  Proc('dmonitoringmodeld', 0.4, msgs=['driverStateV2']),
  Proc('encoderd', 0.23, msgs=[]),
  Proc('mapsd', 0.05, msgs=['mapRenderState']),
  Proc('navmodeld', 0.05, msgs=['navModel']),
]

def send_llk_msg(done):
  # Send liveLocationKalman at 20Hz
  pm = messaging.PubMaster(['liveLocationKalman'])
  while not done.is_set():
    msg = gen_llk()
    pm.send('liveLocationKalman', msg)
    time.sleep(1/20.)


@pytest.mark.tici
class TestPowerDraw(unittest.TestCase):

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
    msg_counts = {}
    for proc in PROCS:
      socks = {msg: messaging.sub_sock(msg) for msg in proc.msgs}
      managed_processes[proc.name].start()
      time.sleep(proc.warmup)
      for sock in socks.values():
        messaging.drain_sock_raw(sock)

      now = get_power(SAMPLE_TIME)
      used[proc.name] = now - prev
      prev = now
      for msg,sock in socks.items():
        msg_counts[msg] = len(messaging.drain_sock_raw(sock))

    done.set()
    manager_cleanup()

    tab = [['process', 'expected (W)', 'measured (W)', '# msgs expected', '# msgs received']]
    for proc in PROCS:
      cur = used[proc.name]
      expected = proc.power
      msgs_received = sum(msg_counts[msg] for msg in proc.msgs)
      msgs_expected = int(sum(SAMPLE_TIME * SERVICE_LIST[msg].frequency for msg in proc.msgs))
      tab.append([proc.name, round(expected, 2), round(cur, 2), msgs_expected, msgs_received])
      with self.subTest(proc=proc.name):
        np.testing.assert_allclose(cur, expected, rtol=proc.rtol, atol=proc.atol)
        np.testing.assert_allclose(msgs_expected, msgs_received, rtol=.02, atol=2)
    print(tabulate(tab))
    print(f"Baseline {baseline:.2f}W\n")


if __name__ == "__main__":
  unittest.main()
