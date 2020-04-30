#!/usr/bin/env python3

from tools.lib.logreader import LogReader
from selfdrive.controls.controlsd import controlsd_thread
from selfdrive.test.profiling.lib import SubMaster, PubMaster, SubSocket, ReplayDone


BASE_URL = "https://commadataci.blob.core.windows.net/openpilotci/"
SEGMENT = "99c94dc769b5d96e|2019-08-03--14-19-59/2"


if __name__ == "__main__":
  segment = SEGMENT.replace('|', '/')
  rlog_url = f"{BASE_URL}{segment}/rlog.bz2"
  msgs = list(LogReader(rlog_url))

  pm = PubMaster(['sendcan', 'controlsState', 'carState', 'carControl', 'carEvents', 'carParams'])
  sm = SubMaster(msgs, 'can', ['thermal', 'health', 'liveCalibration', 'dMonitoringState', 'plan', 'pathPlan', 'model'])
  can_sock = SubSocket(msgs, 'can')

  try:
    controlsd_thread(sm, pm, can_sock)
  except ReplayDone:
    pass
