import unittest
from unittest.mock import Mock, patch

import numpy as np

from openpilot.system import micd


class TestMicd(unittest.TestCase):
  def test_capture_bandwidth_and_analysis_timing(self):
    # Two 50ms callbacks retain a 12kHz tone and produce one 100ms SPL window.
    samples = (0.1 * np.sin(2 * np.pi * 12000 * np.arange(4800) / 48000)).astype(np.float32)
    with patch.object(micd.messaging, 'PubMaster') as publisher, \
         patch.object(micd, 'apply_a_weighting', wraps=micd.apply_a_weighting) as weighting:
      mic = micd.Mic()
      for block in np.split(samples, 2):
        mic.callback(block[:, None], len(block), None, None)

      messages = [call.args[1].rawAudioData for call in publisher.return_value.send.call_args_list]
      assert len(messages) == 2
      assert all(message.sampleRate == 48000 for message in messages)
      captured = np.frombuffer(b''.join(message.data for message in messages), dtype=np.int16)
      np.testing.assert_array_equal(captured, (samples * 32767).astype(np.int16))
      frequencies = np.fft.rfftfreq(len(captured), 1 / messages[0].sampleRate)
      assert frequencies[np.argmax(abs(np.fft.rfft(captured)))] == 12000
      weighting.assert_called_once()
      np.testing.assert_array_equal(weighting.call_args.args[0], samples)
      assert mic.measurements.size == 0
      assert np.isfinite(mic.sound_pressure_level_weighted)

  def test_stream_timing(self):
    with patch.object(micd.messaging, 'PubMaster'):
      mic = micd.Mic()
    sounddevice = Mock()
    mic.get_stream(sounddevice)
    stream = sounddevice.InputStream
    assert stream.call_args.kwargs['channels'] == 1
    assert stream.call_args.kwargs['samplerate'] == 48000
    assert stream.call_args.kwargs['blocksize'] / stream.call_args.kwargs['samplerate'] == 0.05
