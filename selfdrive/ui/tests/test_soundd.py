#!/usr/bin/env python3
import subprocess
import time
import unittest

from cereal import log, car
import cereal.messaging as messaging
from selfdrive.test.helpers import phone_only, with_processes
# TODO: rewrite for unittest
from common.realtime import DT_CTRL
from selfdrive.hardware import HARDWARE

AudibleAlert = car.CarControl.HUDControl.AudibleAlert

SOUNDS = {
  # sound: total writes
  AudibleAlert.none: 0,
  AudibleAlert.engage: 184,
  AudibleAlert.disengage: 186,
  AudibleAlert.refuse: 194,
  AudibleAlert.prompt: 184,
  AudibleAlert.promptRepeat: 487,
  AudibleAlert.promptDistracted: 196,
  AudibleAlert.warningSoft: 471,
  AudibleAlert.warningImmediate: 470,
}

def get_total_writes():
  audio_flinger = subprocess.check_output('dumpsys media.audio_flinger', shell=True, encoding='utf-8').strip()
  write_lines = [l for l in audio_flinger.split('\n') if l.strip().startswith('Total writes')]
  return sum(int(l.split(':')[1]) for l in write_lines)

class TestSoundd(unittest.TestCase):
  def test_sound_card_init(self):
    assert HARDWARE.get_sound_card_online()

  @phone_only
  @with_processes(['soundd'])
  def test_alert_sounds(self):
    pm = messaging.PubMaster(['deviceState', 'controlsState'])

    # make sure they're all defined
    alert_sounds = {v: k for k, v in car.CarControl.HUDControl.AudibleAlert.schema.enumerants.items()}
    diff = set(SOUNDS.keys()).symmetric_difference(alert_sounds.keys())
    assert len(diff) == 0, f"not all sounds defined in test: {diff}"

    # wait for procs to init
    time.sleep(1)

    for sound, expected_writes in SOUNDS.items():
      print(f"testing {alert_sounds[sound]}")
      start_writes = get_total_writes()

      for i in range(int(10 / DT_CTRL)):
        msg = messaging.new_message('deviceState')
        msg.deviceState.started = True
        pm.send('deviceState', msg)

        msg = messaging.new_message('controlsState')
        if i < int(6 / DT_CTRL):
          msg.controlsState.alertSound = sound
          msg.controlsState.alertType = str(sound)
          msg.controlsState.alertText1 = "Testing Sounds"
          msg.controlsState.alertText2 = f"playing {alert_sounds[sound]}"
          msg.controlsState.alertSize = log.ControlsState.AlertSize.mid
        pm.send('controlsState', msg)
        time.sleep(DT_CTRL)

      tolerance = expected_writes / 8
      actual_writes = get_total_writes() - start_writes
      print(f"  expected {expected_writes} writes, got {actual_writes}")
      #assert abs(expected_writes - actual_writes) <= tolerance, f"{alert_sounds[sound]}: expected {expected_writes} writes, got {actual_writes}"

if __name__ == "__main__":
  unittest.main()
