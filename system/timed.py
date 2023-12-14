#!/usr/bin/env python3
import json
import os
import re
import subprocess
import time
from datetime import datetime, timedelta

import requests
from timezonefinder import TimezoneFinder

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.system.hardware import AGNOS
from openpilot.system.version import get_version
from typing import NoReturn

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

  valid_timezones = subprocess.check_output('timedatectl list-timezones', shell=True, encoding='utf8').strip().split('\n')

  while True:
    time.sleep(60)

    is_onroad = not params.get_bool("IsOffroad")
    if is_onroad:
      continue


    # Set timezone with param
    timezone = params.get("Timezone", encoding='utf8')
    if timezone is not None:
      cloudlog.debug("Setting timezone based on param")
      set_timezone(valid_timezones, timezone)
    else:


      # Set timezone with IP lookup
      location = params.get("LastGPSPosition", encoding='utf8')
      if location is None:
        cloudlog.debug("Setting timezone based on IP lookup")
        try:
          r = requests.get("https://ipapi.co/timezone", headers=REQUEST_HEADERS, timeout=10)
          if r.status_code == 200:
            set_timezone(valid_timezones, r.text)
          else:
            cloudlog.error(f"Unexpected status code from api {r.status_code}")
          time.sleep(3600)  # Don't make too many API requests
        except requests.exceptions.RequestException:
          cloudlog.exception("Error getting timezone based on IP")
          continue


      # Set timezone with GPS
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


    # Set time from modem
    try:
      cloudlog.debug("Setting time based on modem")
      output = subprocess.check_output("mmcli -m 0 --time", shell=True).decode()

      date_pattern = r"current: (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}-\d{2})"
      timezone_pattern = r"Timezone \| current: (-?\d+)"

      # Extract date/timezone
      date = re.search(date_pattern, output).group(1) if re.search(date_pattern, output) else None
      timezone = re.search(timezone_pattern, output).group(1) if re.search(timezone_pattern, output) else None

      # Remove timezone from date
      date = date[:-3]

      # Convert to datetime
      date = datetime.strptime(date, "%Y-%m-%dT%H:%M:%S")

      # Fix offset
      date = date - timedelta(days=682, hours=11)

      # Set time
      os.system(f"TZ=UTC date -s '{date.strftime('%Y-%m-%d %H:%M:%S')}'")

    except Exception as e:
        cloudlog.error(f"Error getting time from modem, {e}")
        return None, None



if __name__ == "__main__":
  main()

