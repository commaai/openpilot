#!/usr/bin/env python

# simple service that waits for network access and tries to update every 3 hours

import os
import time
import subprocess

from common.basedir import BASEDIR
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
    r = subprocess.call(NICE_LOW_PRIORITY + ["git", "fetch"])
    cloudlog.info("git fetch: %r", r)
    if r:
      time.sleep(60)
      continue

    # download apks
    r = subprocess.call(NICE_LOW_PRIORITY + [os.path.join(BASEDIR, "apk/external/patcher.py"), "download"])
    cloudlog.info("patcher download: %r", r)

    time.sleep(60*60*3)

if __name__ == "__main__":
  main()

