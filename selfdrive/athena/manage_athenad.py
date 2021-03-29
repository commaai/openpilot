#!/usr/bin/env python3

import time
from multiprocessing import Process

import selfdrive.crash as crash
from common.params import Params
from selfdrive.manager.process import launcher
from selfdrive.swaglog import cloudlog
from selfdrive.version import version, dirty

ATHENA_MGR_PID_PARAM = "AthenadPid"


def main():
  params = Params()
  dongle_id = params.get("DongleId").decode('utf-8')
  cloudlog.bind_global(dongle_id=dongle_id, version=version, dirty=dirty)
  crash.bind_user(id=dongle_id)
  crash.bind_extra(version=version, dirty=dirty)
  crash.install()

  try:
    while 1:
      cloudlog.info("starting athena daemon")
      proc = Process(name='athenad', target=launcher, args=('selfdrive.athena.athenad',))
      proc.start()
      proc.join()
      cloudlog.event("athenad exited", exitcode=proc.exitcode)
      time.sleep(5)
  except Exception:
    cloudlog.exception("manage_athenad.exception")
  finally:
    params.delete(ATHENA_MGR_PID_PARAM)


if __name__ == '__main__':
  main()
