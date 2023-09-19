#!/usr/bin/env python3
import os
import datetime
from panda import Panda

from openpilot.common.time import MIN_DATE

def set_time(logger):
  sys_time = datetime.datetime.today()
  if sys_time > MIN_DATE:
    logger.info("System time valid")
    return

  try:
    ps = Panda.list()
    if len(ps) == 0:
      logger.error("Failed to set time, no pandas found")
      return

    for s in ps:
      with Panda(serial=s) as p:
        if not p.is_internal():
          continue

        # Set system time from panda RTC time
        panda_time = p.get_datetime()
        if panda_time > MIN_DATE:
          logger.info(f"adjusting time from '{sys_time}' to '{panda_time}'")
          os.system(f"TZ=UTC date -s '{panda_time}'")
        break
  except Exception:
    logger.exception("Failed to fetch time from panda")

if __name__ == "__main__":
  import logging
  logging.basicConfig(level=logging.DEBUG)

  set_time(logging)
