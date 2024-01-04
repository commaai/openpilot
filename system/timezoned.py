#!/usr/bin/env python3
import json
import os
import time
import subprocess
from typing import NoReturn

from timezonefinder import TimezoneFinder

from openpilot.common.params import Params
from openpilot.system.hardware import AGNOS
from openpilot.common.swaglog import cloudlog
from openpilot.system.version import get_version

REQUEST_HEADERS = {'User-Agent': "openpilot-" + get_version()}


def set_timezone(valid_timezones, timezone):
  if timezone not in valid_timezones:
    cloudlog.error(f"Timezone not supported {timezone}")
    return

  cloudlog.info(f"Setting timezone to {timezone}")
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

  timezone = params.get("Timezone", encoding='utf8')
  if timezone is not None:
    cloudlog.debug("Setting timezone based on param")
    set_timezone(valid_timezones, timezone)

  while True:
    time.sleep(60)

    location = params.get("LastGPSPosition", encoding='utf8')

    # Find timezone by reverse geocoding the last known gps location
    if location is not None:
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
