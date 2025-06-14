import os
import time
import struct
from openpilot.system.hardware.hw import Paths

WATCHDOG_FN = f"{Paths.shm_path()}/wd_"
_LAST_KICK = 0.0

def kick_watchdog():
  global _LAST_KICK
  current_time = time.monotonic()

  if current_time - _LAST_KICK < 1.0:
    return

  try:
    with open(f"{WATCHDOG_FN}{os.getpid()}", 'wb') as f:
      f.write(struct.pack('<Q', int(current_time * 1e9)))
      f.flush()
    _LAST_KICK = current_time
  except OSError:
    pass
