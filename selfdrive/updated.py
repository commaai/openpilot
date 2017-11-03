#!/usr/bin/env python

# simple service that waits for network access and tries to update every 3 hours

import os
import time
import subprocess

def main(gctx=None):
  if not os.getenv("CLEAN"):
    return

  while True:
    # try network
    r = subprocess.call(["ping", "-W", "4", "-c", "1", "8.8.8.8"])
    if r:
      time.sleep(60)
      continue

    # try fetch
    r = subprocess.call(["nice", "-n", "19", "git", "fetch", "--depth=1"])
    if r:
      time.sleep(60)
      continue

    time.sleep(60*60*3)

if __name__ == "__main__":
    main()
