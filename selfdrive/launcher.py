import importlib
from setproctitle import setproctitle  # pylint: disable=no-name-in-module

import cereal.messaging as messaging
import selfdrive.crash as crash
from selfdrive.swaglog import cloudlog

def launcher(proc):
  try:
    # import the process
    mod = importlib.import_module(proc)

    # rename the process
    setproctitle(proc)

    # create new context since we forked
    messaging.context = messaging.Context()

    # exec the process
    mod.main()
  except KeyboardInterrupt:
    cloudlog.warning("child %s got SIGINT" % proc)
  except Exception:
    # can't install the crash handler becuase sys.excepthook doesn't play nice
    # with threads, so catch it here.
    crash.capture_exception()
    raise
