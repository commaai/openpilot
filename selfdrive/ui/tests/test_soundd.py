import numpy as np
import pytest
import unittest
import time

from cereal import messaging, car
from openpilot.selfdrive.test.helpers import with_processes


AudibleAlert = car.CarControl.HUDControl.AudibleAlert


@pytest.mark.tici
class TestSoundd(unittest.TestCase):
  SOUND_PLAY_TIME = 1
  TOL = 0.3

  SOUNDS_TO_TEST = [AudibleAlert.engage, AudibleAlert.disengage, AudibleAlert.refuse, AudibleAlert.prompt, \
                    AudibleAlert.promptRepeat, AudibleAlert.promptDistracted, AudibleAlert.warningSoft, AudibleAlert.warningImmediate]

  REFERENCE_LEVELS = { # dB above ambient
    AudibleAlert.engage: 11.5,
    AudibleAlert.disengage: 14.5,
    AudibleAlert.refuse: 16.5,
    AudibleAlert.prompt: 13,
    AudibleAlert.promptRepeat: 13,
    AudibleAlert.promptDistracted: 19,
    AudibleAlert.warningSoft: 22.5,
    AudibleAlert.warningImmediate: 15.3,
  }

  @with_processes(["soundd", "micd"])
  def test_sound(self):
    time.sleep(2)

    pm = messaging.PubMaster(['controlsState'])
    sm = messaging.SubMaster(['microphone'])

    self.levels_for_sounds = {}

    for i in range(len(self.SOUNDS_TO_TEST)):
      def send_sound(sound, play_time):
        db_history = []

        play_start = time.monotonic()
        while time.monotonic() - play_start < play_time:
          sm.update(0)

          if sm.updated['microphone']:
            db_history.append(sm["microphone"].soundPressureWeightedDb)

          m1 = messaging.new_message('controlsState')
          m1.controlsState.alertSound = sound

          pm.send('controlsState', m1)
          time.sleep(0.01)

        if sound == AudibleAlert.none:
          self.ambient_db = np.mean(db_history)
        else:
          self.levels_for_sounds[sound] = np.mean(db_history) - self.ambient_db

      send_sound(AudibleAlert.none, self.SOUND_PLAY_TIME*2)
      send_sound(self.SOUNDS_TO_TEST[i], self.SOUND_PLAY_TIME)

    for sound in self.REFERENCE_LEVELS:
      with self.subTest(sound=sound):
        self.assertLess(self.levels_for_sounds[sound], self.REFERENCE_LEVELS[sound] * (1+self.TOL))
        self.assertGreater(self.levels_for_sounds[sound], self.REFERENCE_LEVELS[sound] * (1-self.TOL))


if __name__ == "__main__":
  unittest.main()