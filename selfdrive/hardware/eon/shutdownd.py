#!/usr/bin/env python3
import os
import time
import datetime

from common.params import Params
from selfdrive.hardware.eon.hardware import getprop
from selfdrive.swaglog import cloudlog

def main():
  prev = b""
  params = Params()
  while True:
    with open("/dev/__properties__", 'rb') as f:
      cur = f.read()

    if cur != prev:
      prev = cur

      # 0 for shutdown, 1 for reboot
      prop = getprop("sys.shutdown.requested")
      if prop is not None and len(prop) > 0:
        os.system("pkill -9 loggerd")
        params.put("LastSystemShutdown", f"'{prop}' {datetime.datetime.now()}")
        os.sync()

        time.sleep(120)
        cloudlog.error('shutdown false positive')
        break

    time.sleep(0.1)

if __name__ == "__main__":
  main()
