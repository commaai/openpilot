#!/usr/bin/env python3
import time

import selfdrive.manager as manager
import cereal.messaging as messaging

from cereal import car
from common.realtime import DT_CTRL

def test_sound():

  pm = messaging.PubMaster(['thermal', 'controlsState'])

  alert_sounds = car.CarControl.HUDControl.AudibleAlert.schema.enumerants

  manager.prepare_managed_process('ui')
  manager.start_managed_process('ui')
  
  # TODO: remove dependency on camerad, needed because UI blocks on frames
  manager.prepare_managed_process('camerad')
  manager.start_managed_process('camerad')
  
  # wait for procs to start
  time.sleep(3)

  for _ in range(5):
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


class TestSounds(unittest.TestCase):

  def setUp(self

if __name__ == "__main__":
  try:
    test_sound()
  except KeyboardInterrupt:
    manager.cleanup_all_processes(None, None)
