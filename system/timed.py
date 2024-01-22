#!/usr/bin/env python3
import os
import subprocess
import time
from typing import NoReturn

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


def main() -> NoReturn:
  """
    timed has two responsibilities:
    - getting the current time
    - getting the current timezone

    we get time directly from GPS and lookup timezone from our GPS position.
    AGNOS will also use NTP to update the time.
  """

  # Restore timezone from param
  param_timezone = Params().get("Timezone", encoding='utf8')
  if param_timezone is not None:
    cloudlog.debug("Restoring timezone from param")
    set_timezone(param_timezone)

  tf = TimezoneFinder()
  sm = messaging.SubMaster(['liveLocationKalman'])
  while True:
    sm.update(1000)

    llk = sm['liveLocationKalman']
    if llk.gpsOk and (time.monotonic() - sm.logMonoTime['liveLocationKalman']/1e9) < 0.2:
      # set time
      # TODO: account for unixTimesatmpMillis being a (usually short) time in the past
      cloudlog.debug("Setting time from GPS")
      gps_time = int(llk.unixTimestampMillis / 1000)
      set_time(f"@{gps_time}")

      # timezone
      if len(llk.positionGeodetic) == 3:
        cloudlog.debug("Setting timezone/time based on GPS location")
        gps_timezone = tf.timezone_at(lat=llk.positionGeodetic[0], lng=llk.positionGeodetic[1])
        if gps_timezone is None:
          cloudlog.error(f"No timezone found based on {llk.positionGeodetic}")
        else:
          set_timezone(gps_timezone)


if __name__ == "__main__":
  main()
