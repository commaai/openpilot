#!/usr/bin/env python3

import time
from multiprocessing import Process

from openpilot.common.params import Params
from openpilot.selfdrive.manager.process import launcher
from openpilot.system.swaglog import cloudlog
from openpilot.system.version import get_version, is_dirty

ATHENA_MGR_PID_PARAM = "AthenadPid"


def main():
  params = Params()
  dongle_id = params.get("DongleId").decode('utf-8')
  cloudlog.bind_global(dongle_id=dongle_id, version=get_version(), dirty=is_dirty())

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
