#!/usr/bin/env python

# simple service that waits for network access and tries to update every hour

import time
import subprocess
from selfdrive.swaglog import cloudlog

NICE_LOW_PRIORITY = ["nice", "-n", "19"]
def main(gctx=None):
  while True:
    # try network
    r = subprocess.call(["ping", "-W", "4", "-c", "1", "8.8.8.8"])
    if r:
      time.sleep(60)
      continue

    # download application update
    try:
      r = subprocess.check_output(NICE_LOW_PRIORITY + ["git", "fetch"], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError, e:
      cloudlog.event("git fetch failed",
        cmd=e.cmd,
        output=e.output,
        returncode=e.returncode)
      time.sleep(60)
      continue
    cloudlog.info("git fetch success: %s", r)

    time.sleep(60*60)

if __name__ == "__main__":
  main()

