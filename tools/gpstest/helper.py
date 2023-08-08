import random
import datetime as dt
import subprocess as sp
from typing import Tuple

from laika.downloader import download_nav
from laika.gps_time import GPSTime
from laika.helpers import ConstellationId


def download_rinex():
  # TODO: check if there is a better way to get the full brdc file for LimeGPS
  gps_time = GPSTime.from_datetime(dt.datetime.utcnow())
  utc_time = dt.datetime.utcnow() - dt.timedelta(1)
  gps_time = GPSTime.from_datetime(dt.datetime(utc_time.year, utc_time.month, utc_time.day))
  return download_nav(gps_time, '/tmp/gpstest/', ConstellationId.GPS)


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


def get_random_coords(lat, lon) -> Tuple[float, float, int]:
  # jump around the world
  # max values, lat: -90 to 90, lon: -180 to 180

  lat_add = random.random()*20 + 10
  lon_add = random.random()*20 + 20
  alt = random.randint(-10**3, 4*10**3)

  lat = ((lat + lat_add + 90) % 180) - 90
  lon = ((lon + lon_add + 180) % 360) - 180
  return round(lat, 5), round(lon, 5), alt


def get_continuous_coords(lat, lon, alt) -> Tuple[float, float, int]:
  # continuously move around the world
  lat_add = random.random()*0.01
  lon_add = random.random()*0.01
  alt_add = random.randint(-100, 100)

  lat = ((lat + lat_add + 90) % 180) - 90
  lon = ((lon + lon_add + 180) % 360) - 180
  alt += alt_add
  return round(lat, 5), round(lon, 5), alt
