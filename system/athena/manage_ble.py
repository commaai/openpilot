#!/usr/bin/env python3

import time
from multiprocessing import Process

from openpilot.common.params import Params
from openpilot.system.manager.process import launcher
from openpilot.common.swaglog import cloudlog
from openpilot.system.hardware import HARDWARE, TICI
from openpilot.system.version import get_build_metadata

BLE_MGR_PID_PARAM = "BlePid"


def main():
  if not TICI:
    return

  params = Params()
  dongle_id = params.get("DongleId")
  build_metadata = get_build_metadata()

  cloudlog.bind_global(
    dongle_id=dongle_id,
    version=build_metadata.openpilot.version,
    origin=build_metadata.openpilot.git_normalized_origin,
    branch=build_metadata.channel,
    commit=build_metadata.openpilot.git_commit,
    dirty=build_metadata.openpilot.is_dirty,
    device=HARDWARE.get_device_type(),
  )

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
