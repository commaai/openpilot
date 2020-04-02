#!/usr/bin/env python3
import time
import os
import subprocess
import signal

import cereal.messaging as messaging
from common.basedir import BASEDIR

def send_controls_packet(pm):
  dat = messaging.new_message('controlsState')
  dat.controlsState = {
    "rearViewCam": True,
  }
  pm.send('controlsState', dat)

def main():
  proc = subprocess.Popen(os.path.join(BASEDIR, "selfdrive/camerad/camerad"), cwd=os.path.join(BASEDIR, "selfdrive/camerad"))
  pm = messaging.PubMaster(['controlsState'])

  def terminate(signalNumber, frame):
    print ('got SIGTERM, exiting..')
    proc.send_signal(signal.SIGINT)
    proc.communicate()
    exit()

  signal.signal(signal.SIGTERM, terminate)

  while True:
    send_controls_packet(pm)

if __name__ == '__main__':
  main()