#!/usr/bin/env python3
import argparse
import multiprocessing
import rpyc # pylint: disable=import-error

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


def run_remote_checker(lat, lon, alt, duration, ip_addr) -> bool:
  try:
    con = rpyc.connect(ip_addr, 18861)
    con._config['sync_request_timeout'] = duration+20
  except ConnectionRefusedError:
    print("could not run remote checker is 'rpc_server.py' running???")
    return False

  # blocking call to rpc checker function
  matched, log, info = con.root.exposed_run_checker(lat, lon, alt,
                        timeout=duration,
                        use_laikad=True)
  con.close() # TODO: might wanna fetch more logs here

  print(f"Remote Checker: {log} {info}")
  return matched


def main(ip_addr, continuous_mode, timeout, pos):
  rinex_file = download_rinex()

  lat, lon, alt = pos
  if lat == 0 and lon == 0 and alt == 0:
    lat, lon, alt = get_random_coords(47.2020, 15.7403)

  while True:
    # spoof random location
    spoof_proc = run_lime_gps(rinex_file, f"{lat},{lon},{alt}", timeout)

    # remote checker runs blocking
    if not run_remote_checker(lat, lon, alt, timeout, ip_addr):
      # location could not be matched by ublox module
      pass

    spoof_proc.terminate()

    if continuous_mode:
      lat, lon, alt = get_continuous_coords(lat, lon, alt)
    else:
      lat, lon, alt = get_random_coords(lat, lon)


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
