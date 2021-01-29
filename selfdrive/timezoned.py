#!/usr/bin/env python3
import time
import json
import itertools
import subprocess
from common.params import Params
from selfdrive.swaglog import cloudlog

from selfdrive.hardware import TICI

if TICI:
  from timezonefinder import TimezoneFinder # pylint: disable=import-error


def main():
  if not TICI:
    return

  params = Params()
  tf = TimezoneFinder()

  # Get allowed timezones
  valid_timezones = subprocess.check_output('timedatectl list-timezones', shell=True, encoding='utf8').strip().split('\n')

  time.sleep(1) # Wait for themald to set IsOffroad

  for i in itertools.count():
    # Run on startup, after that once a minute
    if i > 0:
      time.sleep(60)

    is_onroad = params.get("IsOffroad") != b"1"
    if is_onroad:
      continue

    location = params.get("LastGPSPosition", encoding='utf8')

    if location is not None:
      try:
        location = json.loads(location)
      except Exception:
        cloudlog.exception("Error parsing location")
        continue

      timezone = tf.timezone_at(lng=location['longitude'], lat=location['latitude'])
      if timezone is None:
        cloudlog.error(f"No timezone found based on location, {location}")
        continue

      if timezone not in valid_timezones:
        cloudlog.error(f"Timezone not supported {timezone}")
        continue

      cloudlog.info(f"Setting timezone to {timezone}")
      try:
        subprocess.check_call(f'sudo timedatectl set-timezone {timezone}', shell=True)
      except subprocess.CalledProcessError:
        cloudlog.exception(f"Error setting timezone to {timezone}")


if __name__ == "__main__":
  main()
