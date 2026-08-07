import threading

import numpy as np

from openpilot.system.micd import SAMPLE_BUFFER, SAMPLE_RATE, Mic


def test_native_rate_capture_is_published_at_16khz(mocker):
  mic = Mic.__new__(Mic)
  mic.pm = mocker.Mock()
  mic.input_rate = 48000
  mic.lock = threading.Lock()
  mic.measurements = np.empty(0)
  mic.sound_pressure = 0.
  mic.sound_pressure_weighted = 0.
  mic.sound_pressure_level_weighted = 0.
  capture = np.linspace(-0.5, 0.5, SAMPLE_BUFFER * 3, dtype=np.float32).reshape(-1, 1)

  mic.callback(capture, len(capture), None, None)

  service, msg = mic.pm.send.call_args.args
  assert service == "rawAudioData"
  assert msg.rawAudioData.sampleRate == SAMPLE_RATE
  assert len(msg.rawAudioData.data) == SAMPLE_BUFFER * np.dtype(np.int16).itemsize


def test_capture_rate_falls_back_to_device_default(mocker):
  mic = Mic.__new__(Mic)
  sd = mocker.Mock()
  sd.query_devices.return_value = {"name": "USB microphone", "default_samplerate": 44100}
  sd.check_input_settings.side_effect = [ValueError, None]

  assert mic.supported_rate(sd, 4) == 44100
  assert [call.kwargs["samplerate"] for call in sd.check_input_settings.call_args_list] == [16000, 44100]
