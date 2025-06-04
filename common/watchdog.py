import os
import time
import struct
from openpilot.system.hardware.hw import Paths

WATCHDOG_FN = f"{Paths.shm_path()}/wd_"


def kick_watchdog():
  current_time = time.time()
  if not hasattr(kick_watchdog, 'last_kick'):
    kick_watchdog.last_kick = 0

  if current_time - kick_watchdog.last_kick < 1.0:
    return

  try:
    with open(f"{WATCHDOG_FN}{os.getpid()}", 'wb') as f:
      f.write(struct.pack('<Q', int(current_time * 1e9)))
      f.flush()
    kick_watchdog.last_kick = current_time
  except OSError:
    pass
