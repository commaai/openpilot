import numpy as np
import pytest

from openpilot.system.micd import Mic, FFT_SAMPLES, SAMPLE_RATE, calculate_spl


@pytest.mark.parametrize('amplitude', [0.005, 0.5])
def test_voice_boost_preserves_ambient_measurement(mocker, amplitude):
  mocker.patch('openpilot.system.micd.messaging.PubMaster')
  mic = Mic()
  samples = (amplitude * np.sin(2 * np.pi * 1000 * np.arange(FFT_SAMPLES) / SAMPLE_RATE)).astype(np.float32)
  original = samples.copy()
  mic.callback(samples[:, None], len(samples), None, None)
  np.testing.assert_array_equal(samples, original)
  assert mic.sound_pressure == pytest.approx(calculate_spl(original)[0])
  audio = mic.pm.send.call_args.args[1].rawAudioData
  assert audio.sampleRate == SAMPLE_RATE
  pcm = np.frombuffer(audio.data, dtype=np.int16).astype(float) / 32767
  assert np.max(np.abs(pcm)) <= 1
  loud = np.abs(samples) > amplitude / 2
  assert np.all(np.sign(pcm[loud]) == np.sign(samples[loud]))
  if amplitude < 0.01:
    assert np.linalg.norm(pcm) / np.linalg.norm(samples) == pytest.approx(4, rel=0.01)
  else:
    assert np.max(np.abs(pcm)) > 0.9
