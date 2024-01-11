#!/usr/bin/env python3
import os
import time
import subprocess
from typing import NoReturn
from datetime import datetime

from timezonefinder import TimezoneFinder
import cereal.messaging as messaging
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.system.hardware import AGNOS

params = Params()
sm = messaging.SubMaster(['gpsLocation'])


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


def set_time(modem_time):
  try:
    cloudlog.info(f"Setting time to {modem_time}")
    subprocess.run(f"TZ=UTC date -s '{modem_time}'", shell=True, check=True)
  except subprocess.CalledProcessError:
    cloudlog.exception(f"Error setting time to {modem_time}")


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
  while True:
    time.sleep(60)

    # Use timezone param
    timezone = params.get("Timezone", encoding='utf8')
    if timezone is not None:
      cloudlog.debug("Setting timezone based on param")
      set_timezone(timezone)
      continue

    location = params.get("LastGPSPosition", encoding='utf8')

    # Use timezone and time from modem
    if location is None:
      try:
        cloudlog.debug("Setting time based on modem")
        output = get_modem_time_output()

        quarter_hour_offset = calculate_time_zone_offset(output)
        modem_timezone = determine_time_zone(quarter_hour_offset)
        set_timezone(modem_timezone)

        modem_time = parse_and_format_utc_date(output)
        set_time(modem_time)

      except Exception as e:
        cloudlog.error(f"Error getting time from modem, {e}")
      continue

    # Find timezone by reverse geocoding the last known gps location
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
