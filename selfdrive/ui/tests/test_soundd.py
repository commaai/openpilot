import numpy as np
import pytest
import unittest
import time

from cereal import messaging, car
from openpilot.selfdrive.test.helpers import with_processes

AudibleAlert = car.CarControl.HUDControl.AudibleAlert


@pytest.mark.tici
class TestSoundd(unittest.TestCase):
  SOUND_PLAY_TIME = 1 # time to play each sound
  AMBIENT_SOUND_TIME = 10 # time to capture the ambient sound level
  SOUND_GAP_TIME = 5 # time between each test (for ambient sound to level out)
  TOL = 0.2

  SOUNDS_TO_TEST = [AudibleAlert.engage, AudibleAlert.disengage, AudibleAlert.refuse, AudibleAlert.prompt, \
                    AudibleAlert.promptRepeat, AudibleAlert.promptDistracted, AudibleAlert.warningSoft, AudibleAlert.warningImmediate]

  REFERENCE_LEVELS = { # dB above ambient
    AudibleAlert.engage: 2.7,
    AudibleAlert.disengage: 3.0,
    AudibleAlert.refuse: 3.8,
    AudibleAlert.prompt: 3.4,
    AudibleAlert.promptRepeat: 3.0,
    AudibleAlert.promptDistracted: 4.5,
    AudibleAlert.warningSoft: 5.4,
    AudibleAlert.warningImmediate: 4.0,
  }

  @with_processes(["soundd", "micd"])
  def test_sound(self):
    time.sleep(2)

    pm = messaging.PubMaster(['controlsState'])
    sm = messaging.SubMaster(['microphone'])

    levels_for_sounds = {}

    def send_sound(sound, play_time, update_ambient=False):
      db_history = []

      play_start = time.monotonic()
      while time.monotonic() - play_start < play_time:
        sm.update(0)

        if sm.updated['microphone']:
          db_history.append(sm["microphone"].filteredSoundPressureWeightedDb)

        m1 = messaging.new_message('controlsState')
        m1.controlsState.alertSound = sound

        pm.send('controlsState', m1)
        time.sleep(0.01)

      if sound != AudibleAlert.none or update_ambient:
        levels_for_sounds[sound] = np.mean(db_history)
    
    send_sound(AudibleAlert.none, self.AMBIENT_SOUND_TIME, True)

    for i in range(len(self.SOUNDS_TO_TEST)):
      send_sound(self.SOUNDS_TO_TEST[i], self.SOUND_PLAY_TIME)
      send_sound(AudibleAlert.none, self.SOUND_GAP_TIME)

    ambient_level = levels_for_sounds[AudibleAlert.none]
    levels_above_ambient = {k: v - ambient_level for k, v in levels_for_sounds.items()}
    print(ambient_level, levels_above_ambient)

    for sound in self.REFERENCE_LEVELS:
      with self.subTest(sound=sound):
        self.assertGreater(levels_above_ambient[sound], self.REFERENCE_LEVELS[sound] * (1-self.TOL))


if __name__ == "__main__":
  unittest.main()