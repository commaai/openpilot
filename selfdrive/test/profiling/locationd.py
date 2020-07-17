#!/usr/bin/env python3

import cProfile  # pylint: disable=import-error
import pprofile  # pylint: disable=import-error
import pyprof2calltree  # pylint: disable=import-error

from tools.lib.logreader import LogReader
from selfdrive.locationd.locationd import locationd_thread
from selfdrive.test.profiling.lib import SubMaster, PubMaster, ReplayDone

BASE_URL = "https://commadataci.blob.core.windows.net/openpilotci/"

CARS = {
  'toyota': ("77611a1fac303767|2020-02-29--13-29-33/3", "TOYOTA COROLLA TSS2 2019"),
}


def get_inputs(msgs, process):
  sub_socks = ['gpsLocationExternal', 'sensorEvents', 'cameraOdometry', 'liveCalibration', 'carState']
  trigger = 'cameraOdometry'

  sm = SubMaster(msgs, trigger, sub_socks)
  pm = PubMaster()
  return sm, pm


if __name__ == "__main__":
  segment, fingerprint = CARS['toyota']
  segment = segment.replace('|', '/')
  rlog_url = f"{BASE_URL}{segment}/rlog.bz2"
  msgs = list(LogReader(rlog_url))

  # Statistical
  sm, pm = get_inputs(msgs, 'locationd')
  with pprofile.StatisticalProfile()(period=0.00001) as pr:
    try:
      locationd_thread(sm, pm)
    except ReplayDone:
      pass
  pr.dump_stats('cachegrind.out.locationd_statistical')

  # Deterministic
  sm, pm = get_inputs(msgs, 'controlsd')
  with cProfile.Profile() as pr:
    try:
      locationd_thread(sm, pm)
    except ReplayDone:
      pass
  pyprof2calltree.convert(pr.getstats(), 'cachegrind.out.locationd_deterministic')
