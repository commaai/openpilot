#!/usr/bin/env python3

import time
from multiprocessing import Process

from openpilot.common.params import Params
from openpilot.selfdrive.manager.process import launcher
from openpilot.common.swaglog import cloudlog
from openpilot.system.hardware import HARDWARE
from openpilot.system.version import get_version, get_normalized_origin, get_short_branch, get_commit, is_dirty

ATHENA_MGR_PID_PARAM = "AthenadPid"


def main():
  params = Params()
  dongle_id = params.get("DongleId").decode('utf-8')
  cloudlog.bind_global(dongle_id=dongle_id,
                       version=get_version(),
                       origin=get_normalized_origin(),
                       branch=get_short_branch(),
                       commit=get_commit(),
                       dirty=is_dirty(),
                       device=HARDWARE.get_device_type())

  try:
    while 1:
      cloudlog.info("starting athena daemon")
      proc = Process(name='athenad', target=launcher, args=('selfdrive.athena.athenad', 'athenad'))
      proc.start()
      proc.join()
      cloudlog.event("athenad exited", exitcode=proc.exitcode)
      time.sleep(5)
  except Exception:
    cloudlog.exception("manage_athenad.exception")
  finally:
    params.remove(ATHENA_MGR_PID_PARAM)


if __name__ == '__main__':
  main()
