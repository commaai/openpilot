import os
import time
import rpyc
import shutil

from datetime import datetime
from collections import defaultdict

#from common.params import Params
import cereal.messaging as messaging
from selfdrive.manager.process_config import managed_processes
from laika.lib.coordinates import ecef2geodetic

DELTA = 0.001
ALT_DELTA = 30
MATCH_NUM = 10
REPORT_STATS = 10

EPHEM_CACHE = "/data/params/d/LaikadEphemeris"
DOWNLOAD_CACHE = "/tmp/comma_download_cache"

SERVER_LOG_FILE = "/tmp/fuzzy_server.log"
server_log = open(SERVER_LOG_FILE, "w+")

def slog(msg):
  server_log.write(f"{datetime.now().strftime('%H:%M:%S.%f')} | {msg}\n")
  server_log.flush()

def handle_laikad(msg):
  if not hasattr(msg, 'correctedMeasurements'):
    return None

  num_corr = len(msg.correctedMeasurements)
  pos_ecef = msg.positionECEF.value
  pos_geo = []
  if len(pos_ecef) > 0:
    pos_geo = ecef2geodetic(pos_ecef)

  pos_std = msg.positionECEF.std
  pos_valid = msg.positionECEF.valid

  slog(f"{num_corr} {pos_geo} {pos_ecef} {pos_std} {pos_valid}")
  return pos_geo, (num_corr, pos_geo, list(pos_ecef), list(msg.positionECEF.std))

hw_msgs = 0
ephem_msgs = defaultdict(int)
def handle_ublox(msg):
  global hw_msgs, ephem_msgs

  d = msg.to_dict()

  if 'hwStatus2' in d:
    hw_msgs += 1

  if 'ephemeris' in d:
    ephem_msgs[msg.ephemeris.svId] += 1

  num_meas = None
  if 'measurementReport' in d:
    num_meas = msg.measurementReport.numMeas

  return [hw_msgs, ephem_msgs, num_meas]


def start_procs(procs):
  for p in procs:
    managed_processes[p].start()
  time.sleep(1)

def kill_procs(procs):
  for p in procs:
    managed_processes[p].stop()
  time.sleep(3)

def check_alive_procs(procs):
  for p in procs:
    if not managed_processes[p].proc.is_alive():
      return False, "p"
  return True, None


class RemoteCheckerService(rpyc.Service):
  def on_connect(self, conn):
    pass

  def on_disconnect(self, conn):
    pass

  def run_checker(self, slat, slon, salt, sockets, procs, timeout):
    global hw_msgs, ephem_msgs
    hw_msgs = 0
    ephem_msgs = defaultdict(int)

    slog(f"Run test: {slat} {slon} {salt}")

    # quectel_mod = Params().get_bool("UbloxAvailable")

    match_cnt = 0
    stats_laikad = []
    stats_ublox = []

    start_procs(procs)
    sm = messaging.SubMaster(sockets)

    start_time = time.monotonic()
    while True:
      sm.update()

      if sm.updated['ubloxGnss']:
        stats_ublox.append(handle_ublox(sm['ubloxGnss']))

      if sm.updated['gnssMeasurements']:
        pos_geo, stats = handle_laikad(sm['gnssMeasurements'])
        if pos_geo is None or len(pos_geo) == 0:
          continue

        match  = all(abs(g-s) < DELTA for g,s in zip(pos_geo[:2], [slat, slon]))
        match &= abs(pos_geo[2] - salt) < ALT_DELTA
        if match:
          match_cnt += 1
          if match_cnt >= MATCH_NUM:
            return True, "MATCH", f"After: {round(time.monotonic() - start_time, 4)}"

        # keep some stats for error reporting
        stats_laikad.append(stats)

        a, p = check_alive_procs(procs)
        if not a:
          return False, "PROC CRASH", f"{p}"

      if (time.monotonic() - start_time) > timeout:
        kill_procs(procs)

        h = stats_laikad[-REPORT_STATS:]
        if len(h) == 0:
          h = stats_ublox[-REPORT_STATS:]
        return False, "TIMEOUT", f"Last {REPORT_STATS} stats: {h}"


  def exposed_run_checker(self, slat, slon, salt, timeout=180, use_laikad=True):
    try:
      procs = []
      sockets = []

      if use_laikad:
        procs.append("laikad") # pigeond, ubloxd # might wanna keep them running
        sockets += ['ubloxGnss', 'gnssMeasurements']

        if os.path.exists(EPHEM_CACHE):
          os.remove(EPHEM_CACHE)
        shutil.rmtree(DOWNLOAD_CACHE, ignore_errors=True)

      return self.run_checker(slat, slon, salt, sockets, procs, timeout)

    except Exception as e:
      # always make sure processes get killed
      kill_procs(procs)
      return False, "CHECKER CRASHED", f"{str(e)}"


if __name__ == "__main__":
  print(f"Sever Log written to: {SERVER_LOG_FILE}")

  from rpyc.utils.server import ThreadedServer
  t = ThreadedServer(RemoteCheckerService, port=18861)
  t.start()
