import numpy as np

from openpilot.system.webrtc.device.audio import (
  CAPTURE_FRAME_SAMPLES, FRAME_MS, MAX_CAPTURE_SAMPLES, SPEECH_HANGOVER_MS,
  FarEndAudioBuffer, LiveStreamAudioTrack, SpeechProbabilityGate, apply_output_gain,
)


def test_far_end_reference_is_bounded_and_downsampled():
  reference = FarEndAudioBuffer()
  reference.add_48khz(np.arange(600, dtype=np.int16))

  samples = reference.take(200)

  np.testing.assert_array_equal(samples, np.arange(0, 600, 3, dtype=np.int16))


def test_speech_probability_gate_mutes_non_speech_with_hangover():
  gate = SpeechProbabilityGate(threshold=0.1, hangover_ms=SPEECH_HANGOVER_MS)
  samples = np.ones(CAPTURE_FRAME_SAMPLES, dtype=np.int16)

  np.testing.assert_array_equal(gate.process(samples, 0.01), np.zeros_like(samples))
  np.testing.assert_array_equal(gate.process(samples, 0.8), samples)
  for _ in range(SPEECH_HANGOVER_MS // FRAME_MS):
    np.testing.assert_array_equal(gate.process(samples, 0.01), samples)
  np.testing.assert_array_equal(gate.process(samples, 0.01), np.zeros_like(samples))


def test_speech_probability_gate_reset_mutes_immediately():
  gate = SpeechProbabilityGate(threshold=0.1, hangover_ms=SPEECH_HANGOVER_MS)
  samples = np.ones(CAPTURE_FRAME_SAMPLES, dtype=np.int16)
  gate.process(samples, 0.8)

  gate.reset()

  np.testing.assert_array_equal(gate.process(samples, 0.01), np.zeros_like(samples))


def test_capture_backlog_is_bounded_to_realtime(mocker):
  mocker.patch("openpilot.system.webrtc.device.audio.messaging.sub_sock")
  track = LiveStreamAudioTrack(FarEndAudioBuffer())
  msg = mocker.Mock()
  msg.rawAudioData.sampleRate = 16000
  samples = np.arange(MAX_CAPTURE_SAMPLES * 2, dtype=np.int16)
  msg.rawAudioData.data = samples.tobytes()
  mocker.patch("openpilot.system.webrtc.device.audio.messaging.drain_sock", side_effect=([msg], []))

  track._drain_capture()

  assert len(track.capture) == MAX_CAPTURE_SAMPLES
  np.testing.assert_array_equal(track.capture, samples[-MAX_CAPTURE_SAMPLES:])


def test_opus_packets_have_monotonic_timestamps(mocker):
  mocker.patch("openpilot.system.webrtc.device.audio.messaging.sub_sock")
  track = LiveStreamAudioTrack(FarEndAudioBuffer())
  track.capture.extend(np.zeros(CAPTURE_FRAME_SAMPLES * 4, dtype=np.int16))

  track._encode_frame()
  track._encode_frame()
  track._encode_frame()
  track._encode_frame()

  packets = list(track.packets)
  assert track.encoder.bit_rate == 64_000
  assert [packet.pts for packet in packets] == [0, 960]
  assert all(bytes(packet) for packet in packets)


def test_audio_processor_configuration_from_environment(mocker):
  mocker.patch("openpilot.system.webrtc.device.audio.messaging.sub_sock")
  processor = mocker.patch("openpilot.system.webrtc.device.audio.AudioProcessor")
  mocker.patch.dict("os.environ", {
    "WEBRTC_AEC": "0", "WEBRTC_NS": "false", "WEBRTC_HPF": "no", "WEBRTC_AGC": "0",
    "WEBRTC_NS_LEVEL": "3", "WEBRTC_AGC_GAIN_DB": "4.5", "WEBRTC_AGC_MAX_GAIN_DB": "18",
    "WEBRTC_STREAM_DELAY_MS": "90", "WEBRTC_SPEECH_THRESHOLD": "0.25", "WEBRTC_SPEECH_HANGOVER_MS": "450",
    "WEBRTC_OUTPUT_GAIN_DB": "0",
  })

  track = LiveStreamAudioTrack(FarEndAudioBuffer())

  processor.assert_called_once_with(sample_rate=16000, echo_cancellation=False, noise_suppression=False,
                                    high_pass_filter=False, auto_gain_control=False, ns_level=3,
                                    agc_gain_db=4.5, agc_max_gain_db=18., stream_delay_ms=90)
  assert track.speech_gate.threshold == 0.25
  assert track.speech_gate.hangover_frames == 45
  assert track.output_gain_db == 0.


def test_aec_ns_agc_enabled_and_speech_gate_disabled_by_default(mocker):
  mocker.patch("openpilot.system.webrtc.device.audio.messaging.sub_sock")
  processor = mocker.patch("openpilot.system.webrtc.device.audio.AudioProcessor")

  track = LiveStreamAudioTrack(FarEndAudioBuffer())

  assert processor.call_args.kwargs["echo_cancellation"] is True
  assert processor.call_args.kwargs["noise_suppression"] is True
  assert processor.call_args.kwargs["ns_level"] == 2
  assert processor.call_args.kwargs["auto_gain_control"] is True
  assert processor.call_args.kwargs["agc_gain_db"] == 16.
  assert processor.call_args.kwargs["agc_max_gain_db"] == 6.
  assert processor.call_args.kwargs["stream_delay_ms"] == 0
  assert track.output_gain_db == 4.
  assert track.speech_gate_enabled is False


def test_zero_output_gain_is_bit_identical():
  samples = np.array([np.iinfo(np.int16).min, -1, 0, 1, np.iinfo(np.int16).max], dtype=np.int16)

  output = apply_output_gain(samples, 0.)

  assert output is samples
  np.testing.assert_array_equal(output, samples)


def test_output_gain_rounds_symmetrically_and_saturates():
  # 20 * log10(1.5) dB gives an exact 1.5x scale.
  samples = np.array([-32768, -3, -1, 0, 1, 3, 32767], dtype=np.int16)

  output = apply_output_gain(samples, 20. * np.log10(1.5))

  np.testing.assert_array_equal(output, [-32768, -5, -2, 0, 2, 5, 32767])


def test_output_gain_keeps_silence_silent():
  samples = np.zeros(CAPTURE_FRAME_SAMPLES, dtype=np.int16)

  np.testing.assert_array_equal(apply_output_gain(samples, 4.), samples)
