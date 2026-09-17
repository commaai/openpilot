import base64
import io
import wave

import numpy as np
import pytest

from tools.sound_preview.preview import SCENARIOS, simulate


@pytest.mark.parametrize('key', SCENARIOS)
def test_scenario_timing_and_max_volume(key):
  spec = SCENARIOS[key]
  result = simulate(spec['phases'], include_audio=False)
  if spec['expected'] is None:
    assert result['firstMax'] is None
  else:
    assert result['firstMax'] == pytest.approx(spec['expected'], abs=.051)
  assert all(row['volume'] == 1. for row in result['trace'] if row['escalated'])


@pytest.mark.parametrize('alert,filename', [('fcw', 'critical.wav'), ('distracted', 'dm_critical.wav'),
                                           ('aeb', 'silence'), ('stock_aeb', 'silence')])
def test_initial_sound_and_silent_pcm(alert, filename):
  result = simulate([{'alert': alert, 'seconds': 9}])
  assert result['trace'][0]['sound'] == filename
  with wave.open(io.BytesIO(base64.b64decode(result['audio']))) as wav:
    assert wav.getnframes() / wav.getframerate() == 9
    pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2')
  assert bool(np.any(pcm[:8 * 48000])) == (filename != 'silence')
  assert np.any(pcm[8 * 48000:])
  assert result['firstMax'] == 8.
