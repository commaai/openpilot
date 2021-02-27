#!/usr/bin/env python3
import time
import cereal.messaging as messaging
from selfdrive.manager import start_managed_process, kill_managed_process

services = ['deviceState', 'radarState']  # the services needed to be spoofed to start ui offroad
procs = ['camerad', 'ui', 'modeld', 'calibrationd']
[start_managed_process(p) for p in procs]  # start needed processes
pm = messaging.PubMaster(services)

dat_deviceState, dat_radar = [messaging.new_message(s) for s in services]
dat_deviceState.deviceState.started = True

try:
  while True:
    pm.send('deviceState', dat_deviceState)
    pm.send('radarState', dat_radar)
    time.sleep(1 / 100)  # continually send, rate doesn't matter for deviceState
except KeyboardInterrupt:
  [kill_managed_process(p) for p in procs]
