#!/usr/bin/env python3
import time
import subprocess

from cereal import log, car
import cereal.messaging as messaging
from selfdrive.test.helpers import phone_only, with_processes
from common.realtime import DT_CTRL
from selfdrive.hardware import HARDWARE

AudibleAlert = car.CarControl.HUDControl.AudibleAlert

SOUNDS = {
  # sound: total writes
  AudibleAlert.none: 0,
  AudibleAlert.chimeEngage: 173,
  AudibleAlert.chimeDisengage: 173,
  AudibleAlert.chimeError: 173,
  AudibleAlert.chimePrompt: 173,
  AudibleAlert.chimeWarning1: 163,
  AudibleAlert.chimeWarning2: 216,
  AudibleAlert.chimeWarning2Repeat: 470,
  AudibleAlert.chimeWarningRepeat: 468,
}

def get_total_writes():
  audio_flinger = subprocess.check_output('dumpsys media.audio_flinger', shell=True, encoding='utf-8').strip()
  write_lines = [l for l in audio_flinger.split('\n') if l.strip().startswith('Total writes')]
  return sum([int(l.split(':')[1]) for l in write_lines])

@phone_only
def test_sound_card_init():
  assert HARDWARE.get_sound_card_online()


@phone_only
@with_processes(['soundd'])
def test_alert_sounds():
  pm = messaging.PubMaster(['controlsState'])

  # make sure they're all defined
  alert_sounds = {v: k for k, v in car.CarControl.HUDControl.AudibleAlert.schema.enumerants.items()}
  diff = set(SOUNDS.keys()).symmetric_difference(alert_sounds.keys())
  assert len(diff) == 0, f"not all sounds defined in test: {diff}"

  # wait for procs to init
  time.sleep(1)

  for sound, expected_writes in SOUNDS.items():
    print(f"testing {alert_sounds[sound]}")
    start_writes = get_total_writes()

    for _ in range(int(9 / DT_CTRL)):
      msg = messaging.new_message('controlsState')
      msg.controlsState.alertSound = sound
      msg.controlsState.alertType = str(sound)
      msg.controlsState.alertText1 = "Testing Sounds"
      msg.controlsState.alertText2 = f"playing {alert_sounds[sound]}"
      msg.controlsState.alertSize = log.ControlsState.AlertSize.mid
      pm.send('controlsState', msg)
      time.sleep(DT_CTRL)

    tolerance = (expected_writes % 100) * 2
    actual_writes = get_total_writes() - start_writes
    assert abs(expected_writes - actual_writes) <= tolerance, f"{alert_sounds[sound]}: expected {expected_writes} writes, got {actual_writes}"
