#!/usr/bin/env python3
import sys
import time
from typing import List

from common.params import Params
import cereal.messaging as messaging
from selfdrive.manager.process_config import managed_processes

DELTA = 0.001
# assume running openpilot for now
procs: List[str] = []#"ubloxd", "pigeond"]


def main():
  if len(sys.argv) != 4:
    print("args: <latitude> <longitude>")
    return

  quectel_mod = Params().get_bool("UbloxAvailable")
  sol_lat = float(sys.argv[2])
  sol_lon = float(sys.argv[3])

  for p in procs:
    managed_processes[p].start()
    time.sleep(0.5) # give time to startup

  socket = 'gpsLocation' if quectel_mod else 'gpsLocationExternal'
  gps_sock = messaging.sub_sock(socket, timeout=0.1)

  # analyze until the location changed
  while True:
    events = messaging.drain_sock(gps_sock)
    for e in events:
      loc = e.gpsLocation if quectel_mod else e.gpsLocationExternal
      lat = loc.latitude
      lon = loc.longitude

      if abs(lat - sol_lat) < DELTA and abs(lon - sol_lon) < DELTA:
        print("MATCH")
        return

    time.sleep(0.1)

    for p in procs:
      if not managed_processes[p].proc.is_alive():
        print(f"ERROR: '{p}' died")
        return


if __name__ == "__main__":
  main()
  for p in procs:
    managed_processes[p].stop()
