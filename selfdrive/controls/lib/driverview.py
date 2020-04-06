#!/usr/bin/env python3
import os
import subprocess
import multiprocessing
import signal

import cereal.messaging as messaging

from common.basedir import BASEDIR
from selfdrive.controls.lib.gps_helpers import is_rhd_region

def send_controls_packet(pm):
  while True:
    dat = messaging.new_message('controlsState')
    dat.controlsState = {
      "rearViewCam": True,
    }
    pm.send('controlsState', dat)

def send_dmon_packet(pm, d):
  dat = messaging.new_message('dMonitoringState')
  dat.dMonitoringState = {
    "isRHD": d[0],
    "rhdChecked": d[1],
  }
  pm.send('dMonitoringState', dat)

def main():
  pm = messaging.PubMaster(['controlsState', 'dMonitoringState'])
  sm = messaging.SubMaster(['gpsLocation'])
  controls_sender = multiprocessing.Process(target=send_controls_packet, args=[pm])
  controls_sender.start()

  proc_cam = subprocess.Popen(os.path.join(BASEDIR, "selfdrive/camerad/camerad"), cwd=os.path.join(BASEDIR, "selfdrive/camerad"))
  proc_mon = subprocess.Popen(os.path.join(BASEDIR, "selfdrive/modeld/dmonitoringmodeld"), cwd=os.path.join(BASEDIR, "selfdrive/modeld"))
  proc_gps = subprocess.Popen(os.path.join(BASEDIR, "selfdrive/sensord/gpsd"), cwd=os.path.join(BASEDIR, "selfdrive/sensord"))
  
  is_rhd = False
  is_rhd_checked = False

  def terminate(signalNumber, frame):
    print('got SIGTERM, exiting..')
    proc_cam.send_signal(signal.SIGINT)
    proc_mon.send_signal(signal.SIGINT)
    proc_gps.send_signal(signal.SIGINT)
    while proc_cam.poll() is None:
      continue
    controls_sender.terminate()
    exit()

  signal.signal(signal.SIGTERM, terminate)

  os.system("am broadcast -a 'ai.comma.plus.HomeButtonTouchUpInside'"); # auto switch to home to not get stuck

  while True:
    send_dmon_packet(pm, [is_rhd, is_rhd_checked])
    sm.update()

    if not is_rhd_checked and sm.updated['gpsLocation']:
      is_rhd = is_rhd_region(sm['gpsLocation'].latitude, sm['gpsLocation'].longitude)
      is_rhd_checked = True

if __name__ == '__main__':
  main()