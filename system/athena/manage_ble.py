#!/usr/bin/env python3

import time
from multiprocessing import Process

from openpilot.system.manager.process import launcher
from openpilot.common.swaglog import cloudlog


def main():
  try:
    while 1:
      cloudlog.info("starting ble daemon")
      proc = Process(name='ble', target=launcher, args=('system.athena.ble', 'ble'))
      proc.start()
      proc.join()
      cloudlog.event("ble exited", exitcode=proc.exitcode)
      time.sleep(5)
  except Exception:
    cloudlog.exception("manage_ble.exception")


if __name__ == '__main__':
  main()
