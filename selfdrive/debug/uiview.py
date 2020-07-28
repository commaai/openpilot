#!/usr/bin/env python3
import time
import cereal.messaging as messaging
from selfdrive.manager import start_managed_process, kill_managed_process

pm = messaging.PubMaster(['controlsState', 'thermal'])
[start_managed_process(p) for p in ['camerad', 'ui']]
try:
  while True:
    dat_cs, dat_thermal = [messaging.new_message(s) for s in ['controlsState', 'thermal']]
    dat_cs.controlsState.rearViewCam, dat_thermal.thermal.started = False, True
    [pm.send(s, dat) for s, dat in zip(['controlsState', 'thermal'], [dat_cs, dat_thermal])]
    time.sleep(1 / 100)
except KeyboardInterrupt:
  [kill_managed_process(p) for p in ['camerad', 'ui']]
