#!/usr/bin/env python3
# forces a Webbing PMC poll and clears a stuck catBusy: resets the modem, then disables and re-enables
# the active profile. drops cellular for ~1 min.
from openpilot.common.esim.lpa import LPA


if __name__ == '__main__':
  lpa = LPA()
  if not lpa.is_euicc():
    raise SystemExit("no eUICC detected")
  lpa.cycle_profile()
  print("done, PMC should poll within a few minutes")
