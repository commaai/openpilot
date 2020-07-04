#!/usr/bin/env python3
import time

import selfdrive.manager as manager
import cereal.messaging as messaging

from cereal import car
from common.realtime import DT_CTRL

def test_sound():

  pm = messaging.PubMaster(['thermal', 'controlsState'])

  alert_sounds = car.CarControl.HUDControl.AudibleAlert.schema.enumerants

  thermal = messaging.new_message('thermal')
  thermal.thermal.started = True

  manager.prepare_managed_process('ui')
  manager.start_managed_process('ui')
  time.sleep(3)

  for sound in alert_sounds:
    pm.send('thermal', thermal)

    for _ in range(5*DT_CTRL):
      msg = messaging.new_message('controlsState')
      msg.controlsState.enabled = True
      msg.controlsState.active = True
      msg.controlsState.alertSound = sound
      pm.send('controlsState', msg)
      time.sleep(DT_CTRL)

  manager.kill_managed_process('ui')

if __name__ == "__main__":
  test_sound()
