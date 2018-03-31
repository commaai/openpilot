#!/usr/bin/env python

# simple service that waits for network access and tries to update every 3 hours

import os
import time
import subprocess

from common.basedir import BASEDIR
from common.params import Params
from selfdrive.swaglog import cloudlog
from selfdrive.version import dirty

NICE_LOW_PRIORITY = ["nice", "-n", "19"]
def main(gctx=None):
  params = Params()

  while True:
    # try network
    r = subprocess.call(["ping", "-W", "4", "-c", "1", "8.8.8.8"])
    if r:
      time.sleep(60)
      continue

    # If there are modifications we preserve full history
    # and disable update prompting.
    # Otherwise, only store head to save space
    if dirty:
      r = subprocess.call(NICE_LOW_PRIORITY + ["git", "fetch", "--unshallow"])
      is_update_available = False
    else:
      r = subprocess.call(NICE_LOW_PRIORITY + ["git", "fetch", "--depth=1"])
      is_update_available = check_is_update_available()

    is_update_available_str = "1" if is_update_available else "0"
    params.put("IsUpdateAvailable", is_update_available_str)
    cloudlog.info("IsUpdateAvailable: %s", is_update_available_str)
    cloudlog.info("git fetch: %r", r)
    if r:
      time.sleep(60)
      continue

    # download apks
    r = subprocess.call(["nice", "-n", "19", os.path.join(BASEDIR, "apk/external/patcher.py"), "download"])
    cloudlog.info("patcher download: %r", r)

    time.sleep(60*60*3)

def check_is_update_available():
  try:
    local_rev = subprocess.check_output(NICE_LOW_PRIORITY + ["git", "rev-parse", "@"])
    upstream_rev = subprocess.check_output(NICE_LOW_PRIORITY + ["git", "rev-parse", "@{u}"])

    return upstream_rev != local_rev
  except subprocess.CalledProcessError:
    cloudlog.exception("updated: failed to compare local and upstream")

    return False

if __name__ == "__main__":
    main()
