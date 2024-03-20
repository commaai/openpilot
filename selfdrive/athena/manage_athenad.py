#!/usr/bin/env python3

import time
from multiprocessing import Process

from openpilot.common.params import Params
from openpilot.selfdrive.manager.process import launcher
from openpilot.common.swaglog import bind_context, cloudlog


ATHENA_MGR_PID_PARAM = "AthenadPid"


def main():
  params = Params()
  bind_context(cloudlog)

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
