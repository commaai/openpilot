#!/usr/bin/env python3
import time
import cereal.messaging as messaging
from selfdrive.manager import start_managed_process, kill_managed_process

services = ['controlsState', 'thermal', 'radarState']  # the services needed to be spoofed to start ui offroad
procs = ['camerad', 'ui', 'modeld', 'calibrationd']
[start_managed_process(p) for p in procs]  # start needed processes
pm = messaging.PubMaster(services)

dat_cs, dat_thermal, dat_radar = [messaging.new_message(s) for s in services]
dat_cs.controlsState.rearViewCam = False  # ui checks for these two messages
dat_thermal.thermal.started = True

try:
  while True:
    pm.send('controlsState', dat_cs)
    pm.send('thermal', dat_thermal)
    pm.send('radarState', dat_radar)
    time.sleep(1 / 100)  # continually send, rate doesn't matter for thermal
except KeyboardInterrupt:
  [kill_managed_process(p) for p in procs]
