#!/usr/bin/env python3
import os
import datetime
from panda import Panda

from common.time import MIN_DATE
import time

SET_TIME_RETRIES = 10
SET_TIME_DELAY = 0.5

def set_time(logger):
  sys_time = datetime.datetime.today()
  if sys_time > MIN_DATE:
    logger.info("System time valid")
    return

  set_time_count = 0
  time_set = False

  while set_time_count < SET_TIME_RETRIES and not time_set:
    logger.info(os.popen("lsusb").read())
    try:
      ps = Panda.list()
      if len(ps) == 0:
        logger.warning("Failed to set time, no pandas found... retrying...")
      
      else:
        for s in ps:
          with Panda(serial=s) as p:
            if not p.is_internal():
              continue

            # Set system time from panda RTC time
            panda_time = p.get_datetime()
            if panda_time > MIN_DATE:
              logger.info(f"adjusting time from '{sys_time}' to '{panda_time}'")
              os.system(f"TZ=UTC date -s '{panda_time}'")
            
            time_set = True
            break
      
    except Exception:
      logger.warning("Failed to fetch time from panda, retrying...")
    
    time.sleep(SET_TIME_DELAY)
    set_time_count += 1

  logger.error(f"Failed to fetch time from panda after {SET_TIME_RETRIES} retries")


if __name__ == "__main__":
  import logging
  logging.basicConfig(level=logging.DEBUG)

  set_time(logging)
