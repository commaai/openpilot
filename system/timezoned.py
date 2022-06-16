#!/usr/bin/env python3
import json
import os
import time
import subprocess
from typing import NoReturn

import requests
from timezonefinder import TimezoneFinder

from common.params import Params
from system.hardware import AGNOS
from system.swaglog import cloudlog


def set_timezone(valid_timezones, timezone):
  if timezone not in valid_timezones:
    cloudlog.error(f"Timezone not supported {timezone}")
    return

  cloudlog.debug(f"Setting timezone to {timezone}")
  try:
    if AGNOS:
      tzpath = os.path.join("/usr/share/zoneinfo/", timezone)
      subprocess.check_call(f'sudo su -c "ln -snf {tzpath} /data/etc/tmptime && \
                              mv /data/etc/tmptime /data/etc/localtime"', shell=True)
      subprocess.check_call(f'sudo su -c "echo \"{timezone}\" > /data/etc/timezone"', shell=True)
    else:
      subprocess.check_call(f'sudo timedatectl set-timezone {timezone}', shell=True)
  except subprocess.CalledProcessError:
    cloudlog.exception(f"Error setting timezone to {timezone}")


def main() -> NoReturn:
  params = Params()
  tf = TimezoneFinder()

  # Get allowed timezones
  valid_timezones = subprocess.check_output('timedatectl list-timezones', shell=True, encoding='utf8').strip().split('\n')

  while True:
    time.sleep(60)

    is_onroad = not params.get_bool("IsOffroad")
    if is_onroad:
      continue

    # Set based on param
    timezone = params.get("Timezone", encoding='utf8')
    if timezone is not None:
      cloudlog.debug("Setting timezone based on param")
      set_timezone(valid_timezones, timezone)
      continue

    location = params.get("LastGPSPosition", encoding='utf8')

    # Find timezone based on IP geolocation if no gps location is available
    if location is None:
      cloudlog.debug("Setting timezone based on IP lookup")
      try:
        r = requests.get("https://ipapi.co/timezone", timeout=10)
        if r.status_code == 200:
          set_timezone(valid_timezones, r.text)
        else:
          cloudlog.error(f"Unexpected status code from api {r.status_code}")

        time.sleep(3600)  # Don't make too many API requests
      except requests.exceptions.RequestException:
        cloudlog.exception("Error getting timezone based on IP")
        continue

    # Find timezone by reverse geocoding the last known gps location
    else:
      cloudlog.debug("Setting timezone based on GPS location")
      try:
        location = json.loads(location)
      except Exception:
        cloudlog.exception("Error parsing location")
        continue

      timezone = tf.timezone_at(lng=location['longitude'], lat=location['latitude'])
      if timezone is None:
        cloudlog.error(f"No timezone found based on location, {location}")
        continue
      set_timezone(valid_timezones, timezone)


if __name__ == "__main__":
  main()
