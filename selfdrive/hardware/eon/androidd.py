#!/usr/bin/env python3
import time
import psutil

from selfdrive.swaglog import cloudlog

WATCHED_PROCS = ["zygote", "zygote64", "/system/bin/servicemanager", "/system/bin/surfaceflinger", "sleep100"]

def main():
  # TODO: handle modem bootloops
  cloudlog.info("androidd started")

  procs = {}
  while True:
    # check interesting procs
    cp = {p: None for p in WATCHED_PROCS}
    for p in psutil.process_iter():
      cmdline = ''.join(p.cmdline())
      if cmdline in WATCHED_PROCS:
        cp[cmdline] = p.pid

    for p in WATCHED_PROCS:
      if p in procs and cp[p] != procs[p]:
        cloudlog.event("android service pid changed", proc=p, prev=procs[p], cur=cp[p])
      procs[p] = cp[p]

    time.sleep(1)

if __name__ == "__main__":
  main()
