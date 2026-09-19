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



def test_capture_clock_advances_across_mute(mocker):
  from types import SimpleNamespace
  from openpilot.system.webrtc.device.audio import LivestreamAudio, MIC_RATE
  clock = mocker.patch('openpilot.system.webrtc.device.audio.time.monotonic_ns', return_value=100_000_000_000)
  track = mocker.Mock()
  audio = LivestreamAudio(track, mocker.Mock())
  codec = OpusCodec(encoder_rate=MIC_RATE)
  def message():
    return SimpleNamespace(logMonoTime=clock.return_value,
                           rawAudioData=SimpleNamespace(sampleRate=MIC_RATE, data=bytes(codec.encoder_samples * 2)))
  try:
    audio.send_capture(message(), codec)
    track.send_frame.assert_not_called()
    audio.enable(True)
    audio.send_capture(message(), codec)
    first = track.send_frame.call_args.args[1].timestamp
    audio.enable(False)
    clock.return_value += 1_000_000_000
    audio.enable(True)
    audio.send_capture(message(), codec)
    second = track.send_frame.call_args.args[1].timestamp
    assert (second - first) & 0xFFFFFFFF == RATE
    stale = message()
    stale.logMonoTime -= 1_000_000_000
    audio.send_capture(stale, codec)
    assert track.send_frame.call_count == 2
  finally:
    codec.close()


def test_incoming_audio_rejects_duplicates_and_late_packets(mocker):
  from libdatachannel import FrameInfo
  from openpilot.system.webrtc.device.audio import LivestreamAudio
  audio = LivestreamAudio(mocker.Mock(), mocker.Mock())
  for timestamp in (0xFFFFFF00, 704, 704, 0xFFFFFF00, 1664):
    audio.on_frame(b'packet', FrameInfo(timestamp))
  assert audio.received.qsize() == 3
  audio.closed = True
  audio.on_frame(b'packet', FrameInfo(2624))
  assert audio.received.qsize() == 3


def test_audio_routes_through_micd_and_soundd(mocker):
  import asyncio
  from libdatachannel import FrameInfo
  from openpilot.system.webrtc.device.audio import LivestreamAudio, MIC_RATE
  from openpilot.cereal import messaging

  async def exercise():
    publisher = mocker.Mock()
    mocker.patch('openpilot.system.webrtc.device.audio.messaging.PubMaster', return_value=publisher)
    mocker.patch('openpilot.system.webrtc.device.audio.messaging.sub_sock')
    recv = mocker.patch('openpilot.system.webrtc.device.audio.messaging.recv_one_or_none', return_value=None)
    track = mocker.Mock()
    audio = LivestreamAudio(track, mocker.Mock())
    audio.enable(True)
    msg = messaging.new_message('rawAudioData')
    msg.rawAudioData.sampleRate = MIC_RATE
    msg.rawAudioData.data = bytes(MIC_RATE // 20 * 2)
    recv.side_effect = [msg, None, None, None, None, None, None, None, None, None, None, None]
    codec = OpusCodec()
    try:
      audio.on_frame(codec.encode(bytes(SAMPLES * 2)), FrameInfo(0))
    finally:
      codec.close()
    audio.start()
    await asyncio.sleep(0.01)
    await audio.stop()
    assert track.send_frame.call_count == 2
    sent = [call.args[1] for call in publisher.send.call_args_list]
    assert len(sent[0].livestreamAudio.data) == SAMPLES * 2
    assert sent[0].livestreamAudio.sampleRate == RATE
    assert not sent[-1].livestreamAudio.data  # flush at disconnect
    audio.on_error.assert_not_called()

  asyncio.run(exercise())


def test_capture_gain_preserves_quiet_speech_and_limits_peaks(mocker):
  from types import SimpleNamespace
  from openpilot.system.webrtc.device.audio import LivestreamAudio, MIC_RATE
  from openpilot.cereal import messaging
  track = mocker.Mock()
  audio = LivestreamAudio(track, mocker.Mock())
  audio.enable(True)
  codec = SimpleNamespace(encoder_samples=MIC_RATE // 50, encode=mocker.Mock(return_value=b'opus'))
  for value in (0, 128, -128, 32767, -32768):
    audio.capture_eq.reset()
    msg = messaging.new_message('rawAudioData')
    msg.rawAudioData.sampleRate = MIC_RATE
    msg.rawAudioData.data = np.full(codec.encoder_samples, value, dtype=np.int16).tobytes()
    audio.send_capture(msg, codec)
    output = np.frombuffer(codec.encode.call_args.args[0], dtype=np.int16).astype(np.float32)
    assert np.max(np.abs(output)) <= 32767
    if abs(value) == 128:
      np.testing.assert_allclose(output, value * 8, rtol=0.002)
    elif value:
      assert np.all(np.sign(output) == np.sign(value))
    else:
      assert not output.any()
