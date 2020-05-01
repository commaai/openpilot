#!/usr/bin/env python3

import time
import cProfile
import pprofile
import pyprof2calltree

from tools.lib.logreader import LogReader
from selfdrive.controls.controlsd import controlsd_thread
from selfdrive.test.profiling.lib import SubMaster, PubMaster, SubSocket, ReplayDone

BASE_URL = "https://commadataci.blob.core.windows.net/openpilotci/"
SEGMENT = "99c94dc769b5d96e|2019-08-03--14-19-59/2"


def get_inputs(msgs):
  sm = SubMaster(msgs, 'can', ['thermal', 'health', 'liveCalibration', 'dMonitoringState', 'plan', 'pathPlan', 'model'])
  pm = PubMaster(['sendcan', 'controlsState', 'carState', 'carControl', 'carEvents', 'carParams'])
  can_sock = SubSocket(msgs, 'can')
  return sm, pm, can_sock


if __name__ == "__main__":
  segment = SEGMENT.replace('|', '/')
  rlog_url = f"{BASE_URL}{segment}/rlog.bz2"
  msgs = list(LogReader(rlog_url))

  # Statistical
  sm, pm, can_sock = get_inputs(msgs)
  with pprofile.StatisticalProfile()(period=0.00001) as pr:
    try:
      controlsd_thread(sm, pm, can_sock)
    except ReplayDone:
      pass
  pr.dump_stats('cachegrind.out.controlsd_statistical')

  # Deterministic
  sm, pm, can_sock = get_inputs(msgs)
  with cProfile.Profile() as pr:
    try:
      controlsd_thread(sm, pm, can_sock)
    except ReplayDone:
      pass
  pyprof2calltree.convert(pr.getstats(), 'cachegrind.out.controlsd_deterministic')
