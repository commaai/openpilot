#!/usr/bin/env python3
import os
import time
import psutil
from typing import Optional

from common.realtime import set_core_affinity, set_realtime_priority
from selfdrive.swaglog import cloudlog


MAX_MODEM_CRASHES = 3
MODEM_PATH = "/sys/devices/soc/2080000.qcom,mss/subsys5"
WATCHED_PROCS = ["zygote", "zygote64", "/system/bin/servicemanager", "/system/bin/surfaceflinger"]


def get_modem_crash_count() -> Optional[int]:
  try:
    with open(os.path.join(MODEM_PATH, "crash_count")) as f:
      return int(f.read())
  except Exception:
    cloudlog.exception("Error reading modem crash count")
  return None

def get_modem_state() -> str:
  try:
    with open(os.path.join(MODEM_PATH, "state")) as f:
      return f.read().strip()
  except Exception:
    cloudlog.exception("Error reading modem state")
  return ""

def main():
  set_core_affinity(1)
  set_realtime_priority(1)

  procs = {}
  crash_count = 0
  modem_killed = False
  modem_state = "ONLINE"
  while True:
    # check critical android services
    if any(p is None or not p.is_running() for p in procs.values()) or not len(procs):
      cur = {p: None for p in WATCHED_PROCS}
      for p in psutil.process_iter(attrs=['cmdline']):
        cmdline = None if not len(p.info['cmdline']) else p.info['cmdline'][0]
        if cmdline in WATCHED_PROCS:
          cur[cmdline] = p

      if len(procs):
        for p in WATCHED_PROCS:
          if cur[p] != procs[p]:
            cloudlog.event("android service pid changed", proc=p, cur=cur[p], prev=procs[p])
      procs.update(cur)

    # check modem state
    state = get_modem_state()
    if state != modem_state and not modem_killed:
      cloudlog.event("modem state changed", state=state)
    modem_state = state

    # check modem crashes
    cnt = get_modem_crash_count()
    if cnt is not None:
      if cnt > crash_count:
        cloudlog.event("modem crash", count=cnt)
      crash_count = cnt

    # handle excessive modem crashes
    if crash_count > MAX_MODEM_CRASHES and not modem_killed:
      cloudlog.event("killing modem")
      with open("/sys/kernel/debug/msm_subsys/modem", "w") as f:
        f.write("put")
      modem_killed = True

    time.sleep(1)

if __name__ == "__main__":
  main()
