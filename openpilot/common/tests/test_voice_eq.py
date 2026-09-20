import numpy as np
import pytest

from openpilot.common.voice_eq import VoiceEQ


@pytest.mark.parametrize('rate', [16000, 48000])
def test_voice_eq_frequency_response(rate):
  for frequency, expected in [(400, -4), (1000, -3), (3000, 0)] + ([(14000, 2)] if rate == 48000 else []):
    eq = VoiceEQ(rate)
    signal = np.sin(2 * np.pi * frequency * np.arange(rate) / rate)
    filtered = np.concatenate([eq.process(chunk) for chunk in np.array_split(signal, 50)])
    gain = 20 * np.log10(np.linalg.norm(filtered[rate // 2:]) / np.linalg.norm(signal[rate // 2:]))
    assert gain == pytest.approx(expected, abs=0.15)


@pytest.mark.parametrize('rate', [16000, 48000])
def test_voice_eq_packet_boundaries_and_reset(rate):
  signal = np.random.default_rng(42).normal(0, 0.1, rate // 2)
  eq = VoiceEQ(rate)
  whole = eq.process(signal)
  eq.reset()
  chunked = np.concatenate([eq.process(chunk) for chunk in np.array_split(signal, 37)])
  np.testing.assert_allclose(chunked, whole)
  eq.reset()
  assert not eq.process(np.zeros(960)).any()
