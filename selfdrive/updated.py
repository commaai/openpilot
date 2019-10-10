#!/usr/bin/env python3

# simple service that waits for network access and tries to update every hour

import datetime
import subprocess
import time

from common.params import Params
from selfdrive.swaglog import cloudlog

NICE_LOW_PRIORITY = ["nice", "-n", "19"]
def main(gctx=None):
  params = Params()

  while True:
    # try network
    ping_failed = subprocess.call(["ping", "-W", "4", "-c", "1", "8.8.8.8"])
    if ping_failed:
      time.sleep(60)
      continue

    # download application update
    try:
      r = subprocess.check_output(NICE_LOW_PRIORITY + ["git", "fetch"], stderr=subprocess.STDOUT).decode('utf8')
    except subprocess.CalledProcessError as e:
      cloudlog.event("git fetch failed",
        cmd=e.cmd,
        output=e.output,
        returncode=e.returncode)
      time.sleep(60)
      continue
    cloudlog.info("git fetch success: %s", r)

    # Write update available param
    try:
      cur_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).rstrip()
      upstream_hash = subprocess.check_output(["git", "rev-parse", "@{u}"]).rstrip()
      params.put("UpdateAvailable", str(int(cur_hash != upstream_hash)))
    except:
      params.put("UpdateAvailable", "0")

    # Write latest release notes to param
    try:
      r = subprocess.check_output(["git", "--no-pager", "show", "@{u}:RELEASES.md"])
      r = r[:r.find(b'\n\n')] # Slice latest release notes
      params.put("ReleaseNotes", r + b"\n")
    except:
      params.put("ReleaseNotes", "")

    t = datetime.datetime.now().isoformat()
    params.put("LastUpdateTime", t.encode('utf8'))

    time.sleep(60*60)

if __name__ == "__main__":
  main()
