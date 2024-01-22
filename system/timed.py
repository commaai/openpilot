#!/usr/bin/env python3
import os
import time
import json
import subprocess
from typing import NoReturn
from datetime import datetime

from timezonefinder import TimezoneFinder

import cereal.messaging as messaging
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.system.hardware import AGNOS


def set_timezone(timezone):
  valid_timezones = subprocess.check_output('timedatectl list-timezones', shell=True, encoding='utf8').strip().split('\n')
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


def set_time(the_time):
  try:
    cloudlog.info(f"Setting time to {the_time}")
    subprocess.run(f"TZ=UTC date -s '{the_time}'", shell=True, check=True)
  except subprocess.CalledProcessError:
    cloudlog.exception(f"Error setting time to {the_time}")


def get_gps_time(sm):
  try:
    return int(sm['gpsLocation'].unixTimestampMillis / 1000)
  except KeyError:
    cloudlog.exception("Error getting GPS time output")


def get_modem_time_output():
  try:
    return subprocess.check_output("mmcli -m 0 --command AT+QLTS=1", shell=True).decode()
  except subprocess.CalledProcessError:
    cloudlog.exception("Error getting modem time output")


def calculate_time_zone_offset(modem_output):
  return int(modem_output[38:-5])


def determine_time_zone(offset):
  hour_offset = -round(offset / 4)
  return f"Etc/GMT+{hour_offset}" if hour_offset >= 0 else f"Etc/GMT{hour_offset}"


def parse_and_format_utc_date(modem_output):
  return datetime.strptime(modem_output[19:-8], "%Y/%m/%d,%H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")


def main() -> NoReturn:
  """
    timed has two responsibilities:
    - getting the current time
    - getting the current timezone

    we have two sources for time:
    - GPS, this only works while onroad
    - modem, this only works with an active SIM
    AGNOS will also pull time from NTP when available
  """

  params = Params()
  tf = TimezoneFinder()
  sm = messaging.SubMaster(['gpsLocation', 'gpsLocationExternal'])
  while True:
    time.sleep(60)
    sm.update(0)

    # Use timezone from param if set
    param_timezone = params.get("Timezone", encoding='utf8')
    if param_timezone is not None:
      cloudlog.debug("Setting timezone based on param")
      set_timezone(param_timezone)
      # set time manually if using timezone param
      continue

    location = params.get("LastGPSPosition", encoding='utf8')


    # Use timezone and time from modem if GPS not available
    if location is None:
      try:
        cloudlog.debug("Setting timezone/time based on modem")
        output = get_modem_time_output()

        quarter_hour_offset = calculate_time_zone_offset(output)
        modem_timezone = determine_time_zone(quarter_hour_offset)
        modem_time = parse_and_format_utc_date(output)
        set_timezone(modem_timezone)
        set_time(modem_time)

      except Exception as e:
        cloudlog.error(f"Error getting time from modem, {e}")
      continue


    # Use timezone and time from gps location
    cloudlog.debug("Setting timezone/time based on GPS location")
    try:
      location = json.loads(location)
    except Exception:
      cloudlog.exception("Error parsing location")
      continue

    gps_timezone = tf.timezone_at(lng=location['longitude'], lat=location['latitude'])
    if gps_timezone is None:
      cloudlog.error(f"No timezone found based on location, {location}")
      continue
    set_timezone(gps_timezone)

    gps_time = get_gps_time(sm)
    if gps_time == 0:
      cloudlog.error("GPS time not available yet")
      continue
    set_time(f"@{gps_time}")


if __name__ == "__main__":
  main()
