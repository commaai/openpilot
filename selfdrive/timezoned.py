#!/usr/bin/env python3
import time
import json
import subprocess
from common.swaglog import cloudlog
from common.params import Params

from selfdrive.hardware import TICI, HARDWARE


def main():
  if not TICI:
    return

  from timezonefinder import TimezoneFinder

  params = Params()
  while True:
    location = params.get("LastGPSLocation", encoding='utf8')
    if location is not None:
      try:
        location = json.loads(location)
      except json.JSONDecodeError:
        cloudlog.exception("Error parsing location")
        continue


      timezone = tf.timezone_at(lng=location['longitude'], lat=location['latitude'])
      if timezone is None:
        cloudlog.error(f"No timezone found, {location}")
        continue

      cloudlog.info(f"Setting timezone to {timezone}")
      try:
        subprocess.check_call(['sudo', 'timedatectl', 'set-timezone', timezone], shell=True)
      except subprocess.CalledProcesError:
        cloudlog.exception(f"Error setting timezone to {timezone}")


    time.sleep(60)




if __name__ == "__main__":
  main()
