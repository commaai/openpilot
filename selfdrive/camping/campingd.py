#!/usr/bin/env python3
import os
import time
import subprocess

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog

"""
Camping mode Miracast receiver launcher.
- Runs offroad only (manager gating), behind Params key `CampingMode`.
- Launches Miracast receiver for screen mirroring from phones/devices.
"""

def main():
  params = Params()
  cloudlog.event("campingd.start")

  # Launch Miracast receiver using MiracleCast only
  proc = None
  sink_proc = None
  try:
    local_bin = "/data/camping/bin"
    wifid = os.path.join(local_bin, "miracle-wifid")
    sinkctl = os.path.join(local_bin, "miracle-sinkctl")

    if os.path.exists(wifid) and os.access(wifid, os.X_OK):
      proc = subprocess.Popen([wifid])
      cloudlog.event("campingd.receiver", name="miracle-wifid")

      # If sink control exists, run in auto-accept mode to act as a sink
      if os.path.exists(sinkctl) and os.access(sinkctl, os.X_OK):
        # -a: auto-accept; some builds use --autoconnect, but -a is common
        try:
          sink_proc = subprocess.Popen([sinkctl, "-a"])  # non-blocking
          cloudlog.event("campingd.sinkctl", name="miracle-sinkctl", args="-a")
        except Exception:
          cloudlog.exception("campingd.sinkctl_start_failed", error=False)
    else:
      cloudlog.event("campingd.receiver", name="miracast_not_found", path=wifid)

    # heartbeat loop
    while True:
      time.sleep(1.0)
      # allow runtime disable
      if not params.get_bool("CampingMode"):
        cloudlog.event("campingd.stop_param")
        break
  except Exception as e:
    cloudlog.exception("campingd.exception", error=True)
  finally:
    if proc and proc.poll() is None:
      proc.terminate()
      try:
        proc.wait(timeout=2)
      except Exception:
        proc.kill()
    cloudlog.event("campingd.exit")

if __name__ == "__main__":
  main()
