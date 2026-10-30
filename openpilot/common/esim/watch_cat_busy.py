#!/usr/bin/env python3
# probes for catBusy by enabling the already-enabled profile: 0x02 = idle (no-op), 0x05 = proactive session in progress.
# a normal Webbing PMC poll holds catBusy for ~1-2 min, longer than that means it's stuck
import sys
import time
from openpilot.common.esim.lpa import LPA, PROFILE_ERROR_CODES


if __name__ == '__main__':
  interval = int(sys.argv[1]) if len(sys.argv) > 1 else 60
  lpa = LPA()
  if not lpa.is_euicc():
    raise SystemExit("no eUICC detected")
  while True:
    with lpa._acquire_channel():
      iccid = lpa._enabled_iccid()
      if iccid is None:
        raise SystemExit("no enabled profile")
      code = lpa._enable_profile(iccid)
    print(f"{time.strftime('%H:%M:%S')} {PROFILE_ERROR_CODES.get(code, 'ok')} (0x{code:02X})", flush=True)
    time.sleep(interval)
