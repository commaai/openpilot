#!/usr/bin/env python3
import os
import time
import psutil

from common.realtime import set_core_affinity, set_realtime_priority
from selfdrive.swaglog import cloudlog

MAX_MODEM_CRASHES = 3
WATCHED_PROCS = ["zygote", "zygote64", "/system/bin/servicemanager", "/system/bin/surfaceflinger"]


def main():
  set_core_affinity(1)
  set_realtime_priority(1)

  procs = {}
  crash_count = 0
  modem_killed = False
  modem_state = "ONLINE"
  while True:
    # check critical android services
    cp = {p: None for p in WATCHED_PROCS}
    for p in psutil.process_iter():
      cmdline = ''.join(p.cmdline())
      if cmdline in WATCHED_PROCS:
        cp[cmdline] = p.pid

    for p in WATCHED_PROCS:
      if p in procs and cp[p] != procs[p]:
        cloudlog.event("android service pid changed", proc=p, prev=procs[p], cur=cp[p])
      procs[p] = cp[p]

    # check modem crashes
    modem_path = "/sys/devices/soc/2080000.qcom,mss/subsys5"
    try:
      with open(os.path.join(modem_path, "crash_count")) as f:
        cnt = int(f.read())
      if cnt > crash_count:
        cloudlog.event("modem crash", count=cnt)
      crash_count = cnt

    except Exception:
      cloudlog.exception("Error reading modem crash count")
      raise

    # check modem state
    try:
      with open(os.path.join(modem_path, "state")) as f:
        state = f.read().strip()
        if state != modem_state and not modem_killed:
          cloudlog.event("modem state changed", state=state)
        modem_state = state
    except Exception:
      cloudlog.exception("Error reading modem state")

    # handle excessive modem crashes
    if crash_count > MAX_MODEM_CRASHES and not modem_killed:
      cloudlog.event("killing modem")
      os.system("echo put > /sys/kernel/debug/msm_subsys/modem")
      modem_killed = True

    time.sleep(1)

if __name__ == "__main__":
  main()
