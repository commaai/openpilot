#!/usr/bin/env python3
import time
import cereal.messaging as messaging
from selfdrive.manager.process_config import managed_processes

if __name__ == "__main__":
  services = ['controlsState', 'deviceState', 'pandaState', 'carParams']
  procs = ['camerad', 'ui', 'modeld', 'calibrationd']

  for p in procs:
    managed_processes[p].start()

  pm = messaging.PubMaster(services)

  msgs = {s: messaging.new_message(s) for s in services}
  msgs['deviceState'].deviceState.started = True
  msgs['pandaState'].pandaState.ignitionLine = True
  msgs['carParams'].carParams.openpilotLongitudinalControl = True

  try:
    while True:
      time.sleep(1 / 100)  # continually send, rate doesn't matter
      for s in msgs:
        pm.send(s, msgs[s])
  except KeyboardInterrupt:
    for p in procs:
      managed_processes[p].stop()
