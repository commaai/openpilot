#!/usr/bin/env python3
import os
import random
import string
import subprocess
import time
import unittest
from collections import defaultdict
from pathlib import Path

from cereal import log
import cereal.messaging as messaging
from cereal.services import service_list
from common.basedir import BASEDIR
from common.timeout import Timeout
import selfdrive.manager as manager
from selfdrive.loggerd.config import ROOT
from tools.lib.logreader import LogReader

SentinelType =log.Sentinel.SentinelType

CEREAL_SERVICES = [f for f in log.Event.schema.union_fields if f in service_list
                   and service_list[f].should_log and "encode" not in f.lower()]

class TestLoggerd(unittest.TestCase):

  def _get_latest_log_dir(self):
    log_dirs = sorted(Path(ROOT).iterdir(), key=lambda f: f.stat().st_mtime)
    return log_dirs[-1]

  def _get_log_dir(self, x):
    for p in x.split(' '):
      path = Path(p.strip())
      if path.is_dir():
        return path
    return None

  def _gen_bootlog(self):
    with Timeout(5):
      out = subprocess.check_output(["./loggerd", "--bootlog"], cwd=os.path.join(BASEDIR, "selfdrive/loggerd"), encoding='utf-8')

    # check existence
    d = self._get_log_dir(out) 
    path = Path(os.path.join(d, "bootlog.bz2"))
    assert path.is_file(), "failed to create bootlog file"
    return path

  def _check_init_data(self, msgs):
    msg = msgs[0]
    assert msg.which() == 'initData'

  def _check_sentinel(self, msgs, route):
    start_type = SentinelType.startOfRoute if route else SentinelType.startOfSegment
    assert msgs[1].sentinel.type == start_type

    end_type = SentinelType.endOfRoute if route else SentinelType.endOfSegment
    assert msgs[-1].sentinel.type == end_type

  def test_bootlog(self):
    # generate bootlog with fake launch log
    launch_log = ''.join([str(random.choice(string.printable)) for _ in range(100)])
    with open("/tmp/launch_log", "w") as f:
      f.write(launch_log)

    bootlog_path = self._gen_bootlog()
    lr = list(LogReader(str(bootlog_path)))

    # check length
    assert len(lr) == 4 # boot + initData + 2x sentinel
    
    # check initData and sentinel
    self._check_init_data(lr)
    self._check_sentinel(lr, True)

    # check msgs
    bootlog_msgs = [m for m in lr if m.which() == 'boot']
    assert len(bootlog_msgs) == 1

    # sanity check values
    boot = bootlog_msgs.pop().boot
    assert abs(boot.wallTimeNanos - time.time_ns()) < 5*1e9 # within 5s
    assert boot.launchLog == launch_log

    for field, path in [("lastKmsg", "console-ramoops"), ("lastPmsg", "pmsg-ramoops-0")]:
      path = Path(os.path.join("/sys/fs/pstore/", path))
      val = b""
      if path.is_file():
        val = open(path, "rb").read()
      assert getattr(boot, field) == val

  def test_qlog(self):
    qlog_services = [s for s in CEREAL_SERVICES if service_list[s].decimation is not None]
    no_qlog_services = [s for s in CEREAL_SERVICES if service_list[s].decimation is None]

    services = random.sample(qlog_services, random.randint(2, 10)) + \
               random.sample(no_qlog_services, random.randint(2, 10))

    pm = messaging.PubMaster(services)
    time.sleep(3)

    # TODO: loggerd shouldn't require the encoders for the main logging thread
    manager.start_managed_process("camerad")
    services = [s for s in services if s not in ("frame", "frontFrame", "wideFrame", "thumbnail")]

    manager.start_managed_process("loggerd")
    time.sleep(2)

    sent_msgs = defaultdict(list)
    for _ in range(random.randint(2, 10) * 100):
      for s in services:
        try:
          m = messaging.new_message(s)
        except Exception:
          m = messaging.new_message(s, random.randint(2, 10))
        pm.send(s, m)
        sent_msgs[s].append(m)
      time.sleep(0.01)

    time.sleep(5)
    manager.kill_managed_process("loggerd")
    manager.kill_managed_process("camerad")

    qlog_path = os.path.join(self._get_latest_log_dir(), "qlog.bz2")
    lr = list(LogReader(qlog_path))

    # check initData and sentinel
    self._check_init_data(lr)
    self._check_sentinel(lr, True)

    recv_msgs = defaultdict(list)
    for m in lr:
      recv_msgs[m.which()].append(m)

    for s, msgs in sent_msgs.items():
      recv_cnt = len(recv_msgs[s])

      if s in no_qlog_services:
        # check services with no specific decimation aren't in qlog
        assert recv_cnt == 0, f"got {recv_cnt} {s} msgs in qlog"
      else:
        print("checking", s)
        # check logged message count matches decimation
        expected_cnt = len(msgs) // service_list[s].decimation
        print(expected_cnt, recv_cnt)
        #assert recv_cnt == expected_cnt, f"expected {expected_cnt} msgs for {s}, got {recv_cnt}"

  def test_rlog(self):
    services = random.sample(CEREAL_SERVICES, random.randint(5, 10))
    pm = messaging.PubMaster(services)

    time.sleep(2)
    manager.start_managed_process("loggerd")

    # TODO: loggerd shouldn't require the encoders for the main logging thread
    manager.start_managed_process("camerad")
    services = [s for s in services if s not in ("frame", "frontFrame", "wideFrame", "thumbnail")]

    time.sleep(1)

    sent_msgs = defaultdict(list)
    for _ in range(random.randint(2, 10) * 100):
      for s in services:
        try:
          m = messaging.new_message(s)
        except Exception:
          m = messaging.new_message(s, random.randint(2, 10))
        pm.send(s, m)
        sent_msgs[s].append(m)
      time.sleep(0.01)

    time.sleep(1)
    manager.kill_managed_process("loggerd")
    manager.kill_managed_process("camerad")
    
    lr = list(LogReader(os.path.join(self._get_latest_log_dir(), "rlog.bz2")))

    # check initData and sentinel
    self._check_init_data(lr)
    self._check_sentinel(lr, True)

    recv_msgs = defaultdict(list)
    for m in lr:
      recv_msgs[m.which()].append(m)

    for s, msgs in sent_msgs.items():
      recv_cnt = len(recv_msgs[s])

      expected_cnt = len(msgs)
      print(s, expected_cnt, recv_cnt)
      print(msgs[0].to_bytes() == recv_msgs[s][0].as_builder().to_bytes(), msgs[-1].to_bytes() == recv_msgs[s][-1].as_builder().to_bytes())


if __name__ == "__main__":
  unittest.main()
