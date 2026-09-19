import queue

import numpy as np
import pytest
from openpilot.system.webrtc.device.audio import OpusCodec, SAMPLES, RATE, put_latest


def test_opus_round_trip():
  codec = OpusCodec()
  try:
    pcm = (np.sin(np.arange(SAMPLES) * 2 * np.pi * 440 / RATE) * 8000).astype(np.int16).tobytes()
    # Exercise the actual system codec, including its initial lookahead.
    decoded = b"".join(codec.decode(codec.encode(pcm)) for _ in range(5))
    assert len(decoded) == len(pcm) * 5
    assert np.max(np.abs(np.frombuffer(decoded, dtype=np.int16))) > 1000
    with pytest.raises(ValueError):
      codec.encode(bytes(12))
  finally:
    codec.close()
    codec.close()


def test_audio_queue_drops_old_frames():
  buffer = queue.Queue(maxsize=3)
  for value in range(100):
    put_latest(buffer, value)
  assert [buffer.get_nowait() for _ in range(3)] == [97, 98, 99]


def test_audio_opens_on_demand_and_closes(mocker):
  import asyncio
  from openpilot.system.webrtc.device.audio import LivestreamAudio

  async def exercise():
    mic = mocker.MagicMock()
    speaker = mocker.MagicMock()
    open_mic = mocker.patch('sounddevice.RawInputStream', return_value=mic)
    open_speaker = mocker.patch('sounddevice.RawOutputStream', return_value=speaker)
    track = mocker.Mock()
    audio = LivestreamAudio(track, mocker.Mock())
    audio.start()
    await asyncio.sleep(0.02)
    open_mic.assert_not_called()
    open_speaker.assert_not_called()
    audio.enable(True)
    for _ in range(100):
      if mic.start.called:
        break
      await asyncio.sleep(0.005)
    mic.start.assert_called_once()
    audio.enable(False)
    for _ in range(100):
      if mic.close.called:
        break
      await asyncio.sleep(0.005)
    mic.close.assert_called_once()
    codec = OpusCodec()
    try:
      audio.on_frame(codec.encode(bytes(SAMPLES * 2)), None)
    finally:
      codec.close()
    for _ in range(100):
      if speaker.start.called:
        break
      await asyncio.sleep(0.005)
    speaker.start.assert_called_once()
    await audio.stop()
    speaker.close.assert_called_once()
    audio.on_frame(b'late packet', None)
    assert audio.received.empty()
    await audio.stop()

  asyncio.run(exercise())
