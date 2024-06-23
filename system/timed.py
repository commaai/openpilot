#!/usr/bin/env python3
import datetime
import subprocess
import time
from typing import NoReturn

import cereal.messaging as messaging
from openpilot.common.time import system_time_valid
from openpilot.common.swaglog import cloudlog


def set_time(new_time):
  diff = datetime.datetime.now() - new_time
  if diff < datetime.timedelta(seconds=10):
    cloudlog.debug(f"Time diff too small: {diff}")
    return

  cloudlog.debug(f"Setting time to {new_time}")
  try:
    subprocess.run(f"TZ=UTC date -s '{new_time}'", shell=True, check=True)
  except subprocess.CalledProcessError:
    cloudlog.exception("timed.failed_setting_time")


def main() -> NoReturn:
  """
    timed has one responsibility:
    - getting the current time

    GPS directly gives time.
    AGNOS will also use NTP to update the time.
  """

  pm = messaging.PubMaster(['clocks'])
  sm = messaging.SubMaster(['liveLocationKalman'])
  while True:
    sm.update(1000)

    msg = messaging.new_message('clocks')
    msg.valid = system_time_valid()
    msg.clocks.wallTimeNanos = time.time_ns()
    pm.send('clocks', msg)

    llk = sm['liveLocationKalman']
    if not llk.gpsOK or (time.monotonic() - sm.logMonoTime['liveLocationKalman']/1e9) > 0.2:
      continue

    # set time
    # TODO: account for unixTimesatmpMillis being a (usually short) time in the past
    gps_time = datetime.datetime.fromtimestamp(llk.unixTimestampMillis / 1000.)
    set_time(gps_time)

    time.sleep(10)

if __name__ == "__main__":
  main()
