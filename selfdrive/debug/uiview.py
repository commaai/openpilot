#!/usr/bin/env python3
import time
import cereal.messaging as messaging
from selfdrive.manager.process_config import managed_processes


if __name__ == "__main__":
  services = ['controlsState', 'deviceState', 'radarState']  # the services needed to be spoofed to start ui offroad
  procs = ['camerad', 'ui', 'modeld', 'calibrationd']

  for p in procs:
    managed_processes[p].start()

  pm = messaging.PubMaster(services)

  dat_controlsState, dat_deviceState, dat_radar = [messaging.new_message(s) for s in services]
  dat_deviceState.deviceState.started = True

  try:
    while True:
      pm.send('controlsState', dat_controlsState)
      pm.send('deviceState', dat_deviceState)
      pm.send('radarState', dat_radar)
      time.sleep(1 / 100)  # continually send, rate doesn't matter for deviceState
  except KeyboardInterrupt:
    for p in procs:
      managed_processes[p].stop()
