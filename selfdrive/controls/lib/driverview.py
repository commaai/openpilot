#!/usr/bin/env python3
import time

import cereal.messaging as messaging

def send_controls_packet():
  pm = messaging.PubMaster(['controlsState'])
  dat = messaging.new_message('controlsState')
  dat.controlsState = {
    "rearViewCam": True;
  }
  pm.send('controlsState', dat)

def driverview_run(): 
  while True:
    send_controls_packet()
    time.sleep(1)

if __name__ == '__main__':
  driverview_run()