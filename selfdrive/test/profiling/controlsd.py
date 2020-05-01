#!/usr/bin/env python3

import os
import time
import cProfile
import pprofile
import pyprof2calltree

from tools.lib.logreader import LogReader
from selfdrive.controls.controlsd import controlsd_thread
from selfdrive.test.profiling.lib import SubMaster, PubMaster, SubSocket, ReplayDone
from selfdrive.test.process_replay.process_replay import CONFIGS

BASE_URL = "https://commadataci.blob.core.windows.net/openpilotci/"

CARS = {
  'toyota': ("77611a1fac303767|2020-02-29--13-29-33/3", "TOYOTA COROLLA TSS2 2019"),
  'honda': ("99c94dc769b5d96e|2019-08-03--14-19-59/2", "HONDA CIVIC 2016 TOURING"),
}


def get_inputs(msgs, process):
  for config in CONFIGS:
    if config.proc_name == process:
      sub_socks = list(config.pub_sub.keys())
      trigger = sub_socks[0]
      break

  sm = SubMaster(msgs, trigger, sub_socks)
  pm = PubMaster()
  can_sock = SubSocket(msgs, 'can')
  return sm, pm, can_sock


if __name__ == "__main__":
  segment, fingerprint = CARS['toyota']
  segment = segment.replace('|', '/')
  rlog_url = f"{BASE_URL}{segment}/rlog.bz2"
  msgs = list(LogReader(rlog_url))

  os.environ['FINGERPRINT'] = fingerprint

  # Statistical
  sm, pm, can_sock = get_inputs(msgs, 'controlsd')
  with pprofile.StatisticalProfile()(period=0.00001) as pr:
    try:
      controlsd_thread(sm, pm, can_sock)
    except ReplayDone:
      pass
  pr.dump_stats('cachegrind.out.controlsd_statistical')

  # Deterministic
  sm, pm, can_sock = get_inputs(msgs, 'controlsd')
  with cProfile.Profile() as pr:
    try:
      controlsd_thread(sm, pm, can_sock)
    except ReplayDone:
      pass
  pyprof2calltree.convert(pr.getstats(), 'cachegrind.out.controlsd_deterministic')
