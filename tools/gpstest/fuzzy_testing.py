#!/usr/bin/env python3
import sys
import time
import random
import datetime as dt
import subprocess as sp
import multiprocessing
import threading
from typing import Tuple, Any

from laika.downloader import download_nav
from laika.gps_time import GPSTime
from laika.helpers import ConstellationId

cache_dir = '/tmp/gpstest/'


def download_rinex():
  # TODO: check if there is a better way to get the full brdc file for LimeGPS
  gps_time = GPSTime.from_datetime(dt.datetime.utcnow())
  utc_time = dt.datetime.utcnow() - dt.timedelta(1)
  gps_time = GPSTime.from_datetime(dt.datetime(utc_time.year, utc_time.month, utc_time.day))
  return download_nav(gps_time, cache_dir, ConstellationId.GPS)


def exec_LimeGPS_bin(rinex_file: str, location: str, duration: int):
  # this functions should never return, cause return means, timeout is
  # reached or it crashed
  try:
    cmd = ["LimeGPS/LimeGPS", "-e", rinex_file, "-l", location]
    sp.check_output(cmd, timeout=duration)
  except sp.TimeoutExpired:
    print("LimeGPS timeout reached!")
  except Exception as e:
    print(f"LimeGPS crashed: {str(e)}")


def run_lime_gps(rinex_file: str, location: str, duration: int):
  print(f"LimeGPS {location} {duration}")

  p = multiprocessing.Process(target=exec_LimeGPS_bin,
                              args=(rinex_file, location, duration))
  p.start()
  return p


def get_random_coords(lat, lon) -> Tuple[int, int]:
  # jump around the world
  # max values, lat: -90 to 90, lon: -180 to 180

  lat_add = random.random()*20 + 10
  lon_add = random.random()*20 + 20

  lat = ((lat + lat_add + 90) % 180) - 90
  lon = ((lon + lon_add + 180) % 360) - 180
  return round(lat, 5), round(lon, 5)

def get_continuous_coords(lat, lon) -> Tuple[int, int]:
  # continuously move around the world

  lat_add = random.random()*0.01
  lon_add = random.random()*0.01

  lat = ((lat + lat_add + 90) % 180) - 90
  lon = ((lon + lon_add + 180) % 360) - 180
  return round(lat, 5), round(lon, 5)

rc_p: Any = None
def exec_remote_checker(lat, lon, duration, ip_addr):
  global rc_p
  # TODO: good enough for testing
  remote_cmd =  "export PYTHONPATH=/data/pythonpath && "
  remote_cmd += "cd /data/openpilot && "
  remote_cmd += f"timeout {duration} /usr/local/pyenv/shims/python tools/gpstest/remote_checker.py "
  remote_cmd += f"{lat} {lon}"

  ssh_cmd = ["ssh", "-i", "/home/batman/openpilot/xx/phone/key/id_rsa",
             f"comma@{ip_addr}"]
  ssh_cmd += [remote_cmd]

  rc_p = sp.Popen(ssh_cmd, stdout=sp.PIPE)
  rc_p.wait()
  rc_output = rc_p.stdout.read()
  print(f"Checker Result: {rc_output.strip().decode('utf-8')}")


def run_remote_checker(spoof_proc, lat, lon, duration, ip_addr) -> bool:
  checker_thread = threading.Thread(target=exec_remote_checker,
                                    args=(lat, lon, duration, ip_addr))
  checker_thread.start()

  tcnt = 0
  while True:
    if not checker_thread.is_alive():
      # assume this only happens when the signal got matched
      return True

    # the spoofing process has a timeout, kill checker if reached
    if not spoof_proc.is_alive():
      rc_p.kill()
      # spoofing process died, assume timeout
      print("Spoofing process timeout")
      return False

    print(f"Time elapsed: {tcnt}[s]", end = "\r")
    time.sleep(1)
    tcnt += 1


def main():
  if len(sys.argv) < 2:
    print(f"usage: {sys.argv[0]} <ip_addr> [-c]")
  ip_addr = sys.argv[1]

  continuous_mode = False
  if len(sys.argv) == 3 and sys.argv[2] == '-c':
    print("Continuous Mode!")
    continuous_mode = True

  rinex_file = download_rinex()

  duration = 60*3 # max runtime in seconds
  lat, lon = get_random_coords(47.2020, 15.7403)

  while True:
    # spoof random location
    spoof_proc = run_lime_gps(rinex_file, f"{lat},{lon},100", duration)
    start_time = time.monotonic()

    # remote checker runs blocking
    if not run_remote_checker(spoof_proc, lat, lon, duration, ip_addr):
      # location could not be matched by ublox module
      pass

    end_time = time.monotonic()
    spoof_proc.terminate()

    # -1 to count process startup
    print(f"Time to get Signal: {round(end_time - start_time - 1, 4)}")

    if continuous_mode:
      lat, lon = get_continuous_coords(lat, lon)
    else:
      lat, lon = get_random_coords(lat, lon)

if __name__ == "__main__":
  main()
