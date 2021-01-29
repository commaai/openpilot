#!/usr/bin/env python3
import time
import json
import subprocess
from common.params import Params
from selfdrive.swaglog import cloudlog

from selfdrive.hardware import TICI, HARDWARE


def main():
  if not TICI:
    return

  from timezonefinder import TimezoneFinder

  params = Params()
  tf = TimezoneFinder()

  # Get allowed timezones
  valid_timezones = subprocess.check_output('timedatectl list-timezones', shell=True, encoding='utf8').strip().split('\n')

  while True:
    location = params.get("LastGPSPosition", encoding='utf8')
    if location is not None:
      try:
        location = json.loads(location)
      except json.JSONDecodeError:
        cloudlog.exception("Error parsing location")
        continue

      timezone = tf.timezone_at(lng=location['longitude'], lat=location['latitude'])
      if timezone is None:
        cloudlog.error(f"No timezone found based on location, {location}")
        continue

      if timezone not in valid_timezones:
        cloudlog.erro(f"Timezone not supported {timezone}")
        continue

      cloudlog.info(f"Setting timezone to {timezone}")
      try:
        subprocess.check_call(f'sudo timedatectl set-timezone {timezone}', shell=True)
      except subprocess.CalledProcessError:
        cloudlog.exception(f"Error setting timezone to {timezone}")


    time.sleep(60)

if __name__ == "__main__":
  main()
