#!/usr/bin/env python3
import time, subprocess
import cereal.messaging as messaging
from common.basedir import BASEDIR

pm = messaging.PubMaster(['controlsState', 'thermal'])
proc_cam = subprocess.Popen(BASEDIR + "/selfdrive/camerad/camerad", cwd=BASEDIR + "/selfdrive/camerad")
proc_ui = subprocess.Popen(BASEDIR + "/selfdrive/ui/ui", cwd=BASEDIR + "/selfdrive/ui")

while True:
  dat = messaging.new_message('controlsState')
  dat.controlsState.rearViewCam = False
  pm.send('controlsState', dat)
  dat = messaging.new_message('thermal')
  dat.thermal.started = True
  pm.send('thermal', dat)
  time.sleep(1 / 100.)

proc_cam.send_signal(signal.SIGINT)
proc_ui.send_signal(signal.SIGINT)
