#!/usr/bin/env python3

import os
import sys
import cProfile  # pylint: disable=import-error
import pprofile  # pylint: disable=import-error
import pyprof2calltree  # pylint: disable=import-error

from tools.lib.logreader import LogReader
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


def profile(proc, func, car='toyota'):
  segment, fingerprint = CARS[car]
  segment = segment.replace('|', '/')
  rlog_url = f"{BASE_URL}{segment}/rlog.bz2"
  msgs = list(LogReader(rlog_url))

  os.environ['FINGERPRINT'] = fingerprint

  # Statistical
  sm, pm, can_sock = get_inputs(msgs, proc)
  with pprofile.StatisticalProfile()(period=0.00001) as pr:
    try:
      func(sm, pm, can_sock)
    except ReplayDone:
      pass
  pr.dump_stats(f'cachegrind.out.{proc}_statistical')

  # Deterministic
  sm, pm, can_sock = get_inputs(msgs, proc)
  with cProfile.Profile() as pr:
    try:
      func(sm, pm, can_sock)
    except ReplayDone:
      pass
  pyprof2calltree.convert(pr.getstats(), f'cachegrind.out.{proc}_deterministic')


if __name__ == '__main__':
  from selfdrive.controls.controlsd import main as controlsd_thread
  from selfdrive.controls.radard import radard_thread

  procs = {
    'radard': radard_thread,
    'controlsd': controlsd_thread,
  }

  proc = sys.argv[1]
  if proc not in procs:
    print(f"{proc} not available")
    sys.exit(0)
  else:
    profile(proc, procs[proc])
