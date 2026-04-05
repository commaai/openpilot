import numpy as np

from cereal import car
from cereal import messaging
from cereal.messaging import SubMaster, PubMaster
from openpilot.selfdrive.ui.soundd import SELFDRIVE_STATE_TIMEOUT, Soundd, check_selfdrive_timeout_alert, sound_list

import time

AudibleAlert = car.CarControl.HUDControl.AudibleAlert


class TestSoundd:
  def test_check_selfdrive_timeout_alert(self):
    sm = SubMaster(['selfdriveState'])
    pm = PubMaster(['selfdriveState'])

    for _ in range(100):
      cs = messaging.new_message('selfdriveState')
      cs.selfdriveState.enabled = True

      pm.send("selfdriveState", cs)

      time.sleep(0.01)

      sm.update(0)

      assert not check_selfdrive_timeout_alert(sm)

    for _ in range(SELFDRIVE_STATE_TIMEOUT * 110):
      sm.update(0)
      time.sleep(0.01)

    assert check_selfdrive_timeout_alert(sm)

  # TODO: add test with micd for checking that soundd actually outputs sounds


class TestGetSoundData:
  """Tests for get_sound_data boundary conditions (issue #36285)."""

  def _make_soundd_with_test_sound(self, sound_data, alert=AudibleAlert.engage):
    s = Soundd.__new__(Soundd)
    s.current_alert = alert
    s.current_volume = 1.0
    s.current_sound_frame = 0
    s.loaded_sounds = {alert: sound_data}
    return s

  def test_frames_equal_to_sound_length(self):
    sound_data = np.arange(50, dtype=np.float32)
    s = self._make_soundd_with_test_sound(sound_data)
    result = s.get_sound_data(50)
    np.testing.assert_array_equal(result, sound_data)

  def test_frames_less_than_sound_length_single_callback(self):
    sound_data = np.arange(70, dtype=np.float32)
    s = self._make_soundd_with_test_sound(sound_data)
    result = s.get_sound_data(50)
    np.testing.assert_array_equal(result, sound_data[:50])
    assert s.current_sound_frame == 50

  def test_frames_less_than_sound_wraps_correctly(self):
    """Regression test for #36285: second callback must wrap to start of sound, not repeat the tail."""
    sound_data = np.arange(70, dtype=np.float32)
    s = self._make_soundd_with_test_sound(sound_data, alert=AudibleAlert.promptRepeat)

    # First callback: [0..49]
    result1 = s.get_sound_data(50)
    np.testing.assert_array_equal(result1, sound_data[:50])

    # Second callback: should be [50..69, 0..29] (wrap around)
    result2 = s.get_sound_data(50)
    expected = np.concatenate([sound_data[50:70], sound_data[0:30]])
    np.testing.assert_array_equal(result2, expected)

  def test_single_play_stops_after_one_loop(self):
    """Single-play sound should zero-fill after the sound ends."""
    sound_data = np.arange(70, dtype=np.float32)
    s = self._make_soundd_with_test_sound(sound_data)

    result1 = s.get_sound_data(50)
    np.testing.assert_array_equal(result1, sound_data[:50])

    # Second callback: [50..69] then zeros (single play, no repeat)
    result2 = s.get_sound_data(50)
    expected = np.zeros(50, dtype=np.float32)
    expected[:20] = sound_data[50:70]
    np.testing.assert_array_equal(result2, expected)

  def test_frames_larger_than_sound_wraps(self):
    """When frames > sound length, the sound should loop within a single callback."""
    sound_data = np.arange(30, dtype=np.float32)
    s = self._make_soundd_with_test_sound(sound_data, alert=AudibleAlert.promptRepeat)

    result = s.get_sound_data(50)
    expected = np.concatenate([sound_data, sound_data[:20]])
    np.testing.assert_array_equal(result, expected)

  def test_multiple_consecutive_callbacks(self):
    """Multiple callbacks should produce a continuous, correctly-wrapped stream."""
    sound_data = np.arange(70, dtype=np.float32)
    s = self._make_soundd_with_test_sound(sound_data, alert=AudibleAlert.promptRepeat)

    all_samples = []
    for _ in range(10):
      result = s.get_sound_data(50)
      all_samples.append(result)

    # Verify the full stream is a continuous loop of sound_data
    full_stream = np.concatenate(all_samples)  # 500 samples total
    for i in range(len(full_stream)):
      assert full_stream[i] == sound_data[i % len(sound_data)], \
        f"Mismatch at index {i}: got {full_stream[i]}, expected {sound_data[i % len(sound_data)]}"

  def test_no_alert_returns_silence(self):
    s = Soundd.__new__(Soundd)
    s.current_alert = AudibleAlert.none
    s.current_volume = 1.0
    s.current_sound_frame = 0
    s.loaded_sounds = {}

    result = s.get_sound_data(50)
    np.testing.assert_array_equal(result, np.zeros(50, dtype=np.float32))

  def test_volume_scaling(self):
    sound_data = np.ones(50, dtype=np.float32)
    s = self._make_soundd_with_test_sound(sound_data)
    s.current_volume = 0.5
    result = s.get_sound_data(50)
    np.testing.assert_array_almost_equal(result, np.full(50, 0.5, dtype=np.float32))
