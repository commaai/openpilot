#!/usr/bin/env python3
import os
import sys
import random
import datetime as dt
import subprocess as sp
from typing import Tuple

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


def get_random_coords(lat, lon) -> Tuple[int, int]:
  # jump around the world
  # max values, lat: -90 to 90, lon: -180 to 180

  lat_add = random.random()*20 + 10
  lon_add = random.random()*20 + 20

  lat = ((lat + lat_add + 90) % 180) - 90
  lon = ((lon + lon_add + 180) % 360) - 180
  return round(lat, 5), round(lon, 5)


def check_availability() -> bool:
  cmd = ["LimeSuite/builddir/LimeUtil/LimeUtil", "--find"]
  output = sp.check_output(cmd)

  if output.strip() == b"":
    return False

  print(f"Device: {output.strip().decode('utf-8')}")
  return True


def main():
  if not os.path.exists('LimeGPS'):
    print("LimeGPS not found run 'setup.sh' first")
    return

  if not os.path.exists('LimeSuite'):
    print("LimeSuite not found run 'setup.sh' first")
    return

  if not check_availability():
    print("No limeSDR device found!")
    return

  rinex_file = download_rinex()
  lat, lon = get_random_coords(47.2020, 15.7403)

  if len(sys.argv) == 3:
    lat = float(sys.argv[1])
    lon = float(sys.argv[2])

  try:
    print(f"starting LimeGPS, Location: {lat},{lon}")
    cmd = ["LimeGPS/LimeGPS", "-e", rinex_file, "-l", f"{lat},{lon},100"]
    sp.check_output(cmd, stderr=sp.PIPE)
  except KeyboardInterrupt:
    print("stopping LimeGPS")
  except Exception as e:
    out_stderr = e.stderr.decode('utf-8')# pylint:disable=no-member
    if "Device is busy." in out_stderr:
      print("GPS simulation is already running, Device is busy!")
      return

    print(f"LimeGPS crashed: {str(e)}")
    print(f"stderr:\n{e.stderr.decode('utf-8')}")# pylint:disable=no-member

if __name__ == "__main__":
  main()
