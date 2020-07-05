#!/usr/bin/env python3
import time

import selfdrive.manager as manager
import cereal.messaging as messaging

from cereal import car
from common.realtime import DT_CTRL
from selfdrive.test.helpers import with_processes


@with_processes(['ui', 'camerad'])
def test_sound():

  pm = messaging.PubMaster(['thermal', 'controlsState'])

  alert_sounds = car.CarControl.HUDControl.AudibleAlert.schema.enumerants

  # wait for procs to init
  time.sleep(5)

  for sound in alert_sounds:
    print(f"testing {sound}")

    msg = messaging.new_message('thermal')
    msg.thermal.started = True
    pm.send('thermal', msg)

    for _ in range(int(3 / DT_CTRL)):
      msg = messaging.new_message('controlsState')
      msg.controlsState.enabled = True
      msg.controlsState.active = True
      msg.controlsState.alertSound = sound
      msg.controlsState.alertType = str(sound)
      pm.send('controlsState', msg)
      time.sleep(DT_CTRL)

  manager.kill_managed_process('camerad')
  manager.kill_managed_process('ui')


if __name__ == "__main__":
  test_sound()
