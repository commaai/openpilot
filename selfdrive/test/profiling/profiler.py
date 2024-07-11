#!/usr/bin/env python3
import os
import sys
import cProfile
import pprofile
import pyprof2calltree

from openpilot.common.params import Params
from openpilot.tools.lib.logreader import LogReader
from openpilot.selfdrive.test.profiling.lib import SubMaster, PubMaster, SubSocket, ReplayDone
from openpilot.selfdrive.test.process_replay.process_replay import CONFIGS
from openpilot.selfdrive.car.toyota.values import CAR as TOYOTA
from openpilot.selfdrive.car.honda.values import CAR as HONDA
from openpilot.selfdrive.car.volkswagen.values import CAR as VW

BASE_URL = "https://commadataci.blob.core.windows.net/openpilotci/"

CARS = {
  'toyota': ("0982d79ebb0de295|2021-01-03--20-03-36/6", TOYOTA.TOYOTA_RAV4),
  'honda': ("0982d79ebb0de295|2021-01-08--10-13-10/6", HONDA.HONDA_CIVIC),
  "vw": ("ef895f46af5fd73f|2021-05-22--14-06-35/6", VW.AUDI_A3_MK3),
}


def get_inputs(msgs, process, fingerprint):
  for config in CONFIGS:
    if config.proc_name == process:
      sub_socks = list(config.pubs)
      trigger = sub_socks[0]
      break

  # some procs block on CarParams
  for msg in msgs:
    if msg.which() == 'carParams':
      m = msg.as_builder()
      m.carParams.carFingerprint = fingerprint
      Params().put("CarParams", m.carParams.copy().to_bytes())
      break

  sm = SubMaster(msgs, trigger, sub_socks)
  pm = PubMaster()
  if 'can' in sub_socks:
    can_sock = SubSocket(msgs, 'can')
  else:
    can_sock = None
  return sm, pm, can_sock


def profile(proc, func, car='toyota'):
  segment, fingerprint = CARS[car]
  segment = segment.replace('|', '/')
  rlog_url = f"{BASE_URL}{segment}/rlog.bz2"
  msgs = list(LogReader(rlog_url)) * int(os.getenv("LOOP", "1"))

  os.environ['FINGERPRINT'] = fingerprint
  os.environ['SKIP_FW_QUERY'] = "1"
  os.environ['REPLAY'] = "1"

  def run(sm, pm, can_sock):
    try:
      if can_sock is not None:
        func(sm, pm, can_sock)
      else:
        func(sm, pm)
    except ReplayDone:
      pass

  # Statistical
  sm, pm, can_sock = get_inputs(msgs, proc, fingerprint)
  with pprofile.StatisticalProfile()(period=0.00001) as pr:
    run(sm, pm, can_sock)
  pr.dump_stats(f'cachegrind.out.{proc}_statistical')

  # Deterministic
  sm, pm, can_sock = get_inputs(msgs, proc, fingerprint)
  with cProfile.Profile() as pr:
    run(sm, pm, can_sock)
  pyprof2calltree.convert(pr.getstats(), f'cachegrind.out.{proc}_deterministic')


if __name__ == '__main__':
  from openpilot.selfdrive.controls.controlsd import main as controlsd_thread
  from openpilot.selfdrive.locationd.paramsd import main as paramsd_thread
  from openpilot.selfdrive.controls.plannerd import main as plannerd_thread

  procs = {
    'controlsd': controlsd_thread,
    'paramsd': paramsd_thread,
    'plannerd': plannerd_thread,
  }

  proc = sys.argv[1]
  if proc not in procs:
    print(f"{proc} not available")
    sys.exit(0)
  else:
    profile(proc, procs[proc])
