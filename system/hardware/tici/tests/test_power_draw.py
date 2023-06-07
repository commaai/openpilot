#!/usr/bin/env python3
import unittest
import time
import math
import queue
import threading
from dataclasses import dataclass, field
from tabulate import tabulate
from typing import List

import cereal.messaging as messaging
from cereal.services import service_list
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
  msgs: List[str] = field(default_factory=list)

PROCS = [
  Proc('camerad', 2.1),
  Proc('modeld', 0.93, atol=0.2, msgs=['modelV2']),
  Proc('dmonitoringmodeld', 0.4, msgs=['driverStateV2']),
  Proc('encoderd', 0.23),
  Proc('mapsd', 0.05, msgs=['mapRenderState']),
  Proc('navmodeld', 0.05, msgs=['navModel']),
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

def get_msg_count(sm, listen_q, msg_count_q, done):
  while (msgs := listen_q.get()) is not None:
    done.clear()
    msg_count = {k:0 for k in msgs}
    while not done.is_set():
      sm.update(1000)
      for msg in msgs:
        if sm.updated[msg]:
          msg_count[msg] += 1
    msg_count_q.put(msg_count)


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
    pub_done = threading.Event()
    pub_thread = threading.Thread(target=send_llk_msg, args=(pub_done,), daemon=True)
    pub_thread.start()

    sm = messaging.SubMaster([msg for proc in PROCS for msg in proc.msgs])
    listen_q = queue.Queue()
    msg_count_q = queue.Queue()
    sub_done = threading.Event()
    sub_thread = threading.Thread(target=get_msg_count, args=(sm, listen_q, msg_count_q, sub_done), daemon=True)
    sub_thread.start()

    baseline = get_power()
    prev = baseline
    used = {}
    msg_counts = {}
    for proc in PROCS:
      listen_q.put(proc.msgs)
      managed_processes[proc.name].start()
      time.sleep(proc.warmup)

      now = get_power(8)
      used[proc.name] = now - prev
      prev = now

      sub_done.set()
      msg_counts.update(msg_count_q.get())

    pub_done.set()
    listen_q.put(None)
    manager_cleanup()

    tab = [['process', 'expected (W)', 'measured (W)']]
    msgtab = [['message', '# expected', '# received']]
    for proc in PROCS:
      cur = used[proc.name]
      expected = proc.power
      tab.append([proc.name, round(expected, 2), round(cur, 2)])
      with self.subTest(proc=proc.name):
        self.assertTrue(math.isclose(cur, expected, rel_tol=proc.rtol, abs_tol=proc.atol))
      for msg in proc.msgs:
        received = msg_counts[msg]
        expected = (proc.warmup + 8) * service_list[msg].frequency
        msgtab.append([msg, int(expected), received])
        with self.subTest(proc=proc.name, msg=msg):
          self.assertTrue(math.isclose(expected, received, rel_tol=.15))
    print(tabulate(msgtab))
    print(tabulate(tab))
    print(f"Baseline {baseline:.2f}W\n")


if __name__ == "__main__":
  unittest.main()
