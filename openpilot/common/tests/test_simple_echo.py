import numpy as np
import pytest

from openpilot.common.simple_echo import SimpleEchoCanceller
from openpilot.common.voice_eq import VoiceEQ


def reference_audio():
  return np.random.default_rng(12).normal(0, 0.06, 48000).astype(np.float32)


@pytest.mark.parametrize('polarity', [-1, 1])
def test_delayed_echo_is_removed_with_nearby_voice_preserved(polarity):
  source = reference_audio()
  rendered = VoiceEQ(48000).process(source)
  echo = SimpleEchoCanceller()
  for offset in range(0, 48000, 960):
    echo.push(source[offset:offset + 960], 10 + offset / 48000)
  # 137 ms of output/capture delay, including a non-decimated sample offset.
  delay = 6577
  start = 40000
  mic = rendered[start - delay:start - delay + 2400] * 0.4 * polarity
  near = np.random.default_rng(13).normal(0, 0.005, len(mic)).astype(np.float32)
  cleaned = echo.process(mic + near, 10 + (start + 2400) / 48000)
  assert np.linalg.norm(cleaned - near) < np.linalg.norm(mic) * 0.03


def test_unrelated_voice_silence_and_stale_reference_pass_through():
  echo = SimpleEchoCanceller()
  source = reference_audio()
  near = np.random.default_rng(14).normal(0, 0.05, 2400).astype(np.float32)
  np.testing.assert_array_equal(echo.process(near, 10), near)
  echo.push(source, 10)
  np.testing.assert_array_equal(echo.process(near, 10.4), near)
  np.testing.assert_array_equal(echo.process(near, 12), near)
  np.testing.assert_array_equal(echo.process(np.zeros(2400, dtype=np.float32), 10.4), 0)
  echo.reset()
  assert not len(echo.reference)


def test_reference_memory_is_bounded_and_gap_resets_history():
  echo = SimpleEchoCanceller()
  for offset in range(100):
    echo.push(np.zeros(960, dtype=np.float32), 10 + offset * 0.02)
  assert len(echo.reference) <= 48000
  echo.push(np.ones(960, dtype=np.float32) * 0.01, 20)
  assert len(echo.reference) == 960
  assert echo.start == 20
