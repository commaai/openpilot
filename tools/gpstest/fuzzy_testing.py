#!/usr/bin/env python3
import argparse
import multiprocessing
import rpyc
from collections import defaultdict

from helper import download_rinex, exec_LimeGPS_bin
from helper import get_random_coords, get_continuous_coords

#------------------------------------------------------------------------------
# this script is supposed to run on HOST PC
# limeSDR is unreliable via c3 USB
#------------------------------------------------------------------------------


def run_lime_gps(rinex_file: str, location: str, timeout: int):
  # needs to run longer than the checker
  timeout += 10
  print(f"LimeGPS {location} {timeout}")
  p = multiprocessing.Process(target=exec_LimeGPS_bin,
                              args=(rinex_file, location, timeout))
  p.start()
  return p

con = None
def run_remote_checker(lat, lon, alt, duration, ip_addr):
  global con
  try:
    con = rpyc.connect(ip_addr, 18861)
    con._config['sync_request_timeout'] = duration+20
  except ConnectionRefusedError:
    print("could not run remote checker is 'rpc_server.py' running???")
    return False, None, None

  matched, log, info = con.root.exposed_run_checker(lat, lon, alt,
                        timeout=duration)
  con.close() # TODO: might wanna fetch more logs here
  con = None

  print(f"Remote Checker: {log} {info}")
  return matched, log, info


stats = defaultdict(int) # type: ignore
keys = ['success', 'failed', 'ublox_fail', 'proc_crash', 'checker_crash']

def print_report():
  print("\nFuzzy testing report summary:")
  for k in keys:
    print(f"  {k}: {stats[k]}")


def update_stats(matched, log, info):
  if matched:
    stats['success'] += 1
    return

  stats['failed'] += 1
  if log == "PROC CRASH":
    stats['proc_crash'] += 1
  if log == "CHECKER CRASHED":
    stats['checker_crash'] += 1
  if log == "TIMEOUT":
    stats['ublox_fail'] += 1


def main(ip_addr, continuous_mode, timeout, pos):
  rinex_file = download_rinex()

  lat, lon, alt = pos
  if lat == 0 and lon == 0 and alt == 0:
    lat, lon, alt = get_random_coords(47.2020, 15.7403)

  try:
    while True:
      # spoof random location
      spoof_proc = run_lime_gps(rinex_file, f"{lat},{lon},{alt}", timeout)

      # remote checker execs blocking
      matched, log, info = run_remote_checker(lat, lon, alt, timeout, ip_addr)
      update_stats(matched, log, info)
      spoof_proc.terminate()
      spoof_proc = None

      if continuous_mode:
        lat, lon, alt = get_continuous_coords(lat, lon, alt)
      else:
        lat, lon, alt = get_random_coords(lat, lon)
  except KeyboardInterrupt:
    if spoof_proc is not None:
      spoof_proc.terminate()

    if con is not None and not con.closed:
      con.root.exposed_kill_procs()
      con.close()

    print_report()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Fuzzy test GPS stack with random locations.")
  parser.add_argument("ip_addr", type=str)
  parser.add_argument("-c", "--contin", type=bool, nargs='?', default=False, help='Continous location change')
  parser.add_argument("-t", "--timeout", type=int, nargs='?', default=180, help='Timeout to get location')

  # for replaying a location
  parser.add_argument("lat", type=float, nargs='?', default=0)
  parser.add_argument("lon", type=float, nargs='?', default=0)
  parser.add_argument("alt", type=float, nargs='?', default=0)
  args = parser.parse_args()
  main(args.ip_addr, args.contin, args.timeout, (args.lat, args.lon, args.alt))
