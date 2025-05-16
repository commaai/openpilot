from collections import defaultdict, deque
import pytest
import time
import numpy as np
from dataclasses import dataclass
from tabulate import tabulate

import cereal.messaging as messaging
from cereal.services import SERVICE_LIST
from opendbc.car.car_helpers import get_demo_car_params
from openpilot.common.mock import mock_messages
from openpilot.common.params import Params
from openpilot.system.hardware.tici.power_monitor import get_power
from openpilot.system.manager.process_config import managed_processes
from openpilot.system.manager.manager import manager_cleanup

SAMPLE_TIME = 8       # seconds to sample power
MAX_WARMUP_TIME = 30  # seconds to wait for SAMPLE_TIME consecutive valid samples

@dataclass
class Proc:
  procs: list[str]
  power: float
  msgs: list[str]
  rtol: float = 0.05
  atol: float = 0.12

  @property
  def name(self):
    return '+'.join(self.procs)


PROCS = [
  Proc(['camerad'], 1.75, msgs=['roadCameraState', 'wideRoadCameraState', 'driverCameraState']),
  Proc(['modeld'], 1.12, atol=0.2, msgs=['modelV2']),
  Proc(['dmonitoringmodeld'], 1.4, msgs=['driverStateV2']),
  Proc(['encoderd'], 0.23, msgs=[]),
]


@pytest.mark.tici
class TestPowerDraw:

  def setup_method(self):
    Params().put("CarParams", get_demo_car_params().to_bytes())

    # wait a bit for power save to disable
    time.sleep(5)

  def teardown_method(self):
    manager_cleanup()

  def get_expected_messages(self, proc):
    return int(sum(SAMPLE_TIME * SERVICE_LIST[msg].frequency for msg in proc.msgs))

  def valid_msg_count(self, proc, msg_counts):
    msgs_received = sum(msg_counts[msg] for msg in proc.msgs)
    msgs_expected = self.get_expected_messages(proc)
    return np.isclose(msgs_expected, msgs_received, rtol=.02, atol=2)

  def valid_power_draw(self, proc, used):
    return np.isclose(used, proc.power, rtol=proc.rtol, atol=proc.atol)

  def tabulate_msg_counts(self, msgs_and_power):
    msg_counts = defaultdict(int)
    for _, counts in msgs_and_power:
      for msg, count in counts.items():
        msg_counts[msg] += count
    return msg_counts

  def get_power_with_warmup_for_target(self, proc, prev):
    socks = {msg: messaging.sub_sock(msg) for msg in proc.msgs}
    for sock in socks.values():
      messaging.drain_sock_raw(sock)

    msgs_and_power = deque([], maxlen=SAMPLE_TIME)

    start_time = time.monotonic()

    while (time.monotonic() - start_time) < MAX_WARMUP_TIME:
      power = get_power(1)
      iteration_msg_counts = {}
      for msg,sock in socks.items():
        iteration_msg_counts[msg] = len(messaging.drain_sock_raw(sock))
      msgs_and_power.append((power, iteration_msg_counts))

      if len(msgs_and_power) < SAMPLE_TIME:
        continue

      msg_counts = self.tabulate_msg_counts(msgs_and_power)
      now = np.mean([m[0] for m in msgs_and_power])

      if self.valid_msg_count(proc, msg_counts) and self.valid_power_draw(proc, now - prev):
        break

    return now, msg_counts, time.monotonic() - start_time - SAMPLE_TIME

  @mock_messages(['livePose'])
  def test_camera_procs(self, subtests):
    baseline = get_power()

    prev = baseline
    used = {}
    warmup_time = {}
    msg_counts = {}

    for proc in PROCS:
      for p in proc.procs:
        managed_processes[p].start()
      now, local_msg_counts, warmup_time[proc.name] = self.get_power_with_warmup_for_target(proc, prev)
      msg_counts.update(local_msg_counts)

      used[proc.name] = now - prev
      prev = now

    manager_cleanup()

    tab = [['process', 'expected (W)', 'measured (W)', '# msgs expected', '# msgs received', "warmup time (s)"]]
    for proc in PROCS:
      cur = used[proc.name]
      expected = proc.power
      msgs_received = sum(msg_counts[msg] for msg in proc.msgs)
      tab.append([proc.name, round(expected, 2), round(cur, 2), self.get_expected_messages(proc), msgs_received, round(warmup_time[proc.name], 2)])
      with subtests.test(proc=proc.name):
        assert self.valid_msg_count(proc, msg_counts), f"expected {self.get_expected_messages(proc)} msgs, got {msgs_received} msgs"
        assert self.valid_power_draw(proc, cur), f"expected {expected:.2f}W, got {cur:.2f}W"
    print(tabulate(tab))
    print(f"Baseline {baseline:.2f}W\n")
