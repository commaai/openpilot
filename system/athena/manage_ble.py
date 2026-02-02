#!/usr/bin/env python3

import os
import time
from multiprocessing import Process

from openpilot.common.params import Params
from openpilot.system.manager.process import launcher
from openpilot.common.swaglog import cloudlog

BLE_MGR_PID_PARAM = "BlePid"


def main():
  # BLE hardware only exists on TICI (comma 3/3X)
  if not os.path.exists("/dev/ttyHS1"):
    return

  params = Params()
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
  finally:
    params.remove(BLE_MGR_PID_PARAM)


if __name__ == '__main__':
  main()
