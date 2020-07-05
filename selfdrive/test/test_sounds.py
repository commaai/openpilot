#!/usr/bin/env python3
import time
import nose

from cereal import car
import cereal.messaging as messaging
from common.android import get_sound_card_online
from common.realtime import DT_CTRL
from selfdrive.test.helpers import phone_only, with_processes


@phone_only
def test_sound_card_init():
  assert get_sound_card_online()


@phone_only
@with_processes(['ui', 'camerad'])
def test_alert_sounds():

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


if __name__ == "__main__":
  nose.run()
