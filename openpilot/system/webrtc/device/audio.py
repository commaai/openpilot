import asyncio
import fractions
import time

from aiortc.mediastreams import MediaStreamError, MediaStreamTrack
from av import AudioFrame
import numpy as np

from openpilot.cereal import messaging
from openpilot.system import micd


WEBRTC_AUDIO_SERVICE = "webrtcAudioData"
WEBRTC_AUDIO_CHANNELS = 1
WEBRTC_AUDIO_BYTES_PER_SAMPLE = 2
WEBRTC_AUDIO_PTIME = 0.020
WEBRTC_AUDIO_TARGET_BUFFER = 0.020
WEBRTC_AUDIO_INITIAL_BUFFER = WEBRTC_AUDIO_PTIME + WEBRTC_AUDIO_TARGET_BUFFER
WEBRTC_AUDIO_MAX_BUFFER = 0.060
WEBRTC_AUDIO_REFERENCE_MAX_BUFFER = 0.200
WEBRTC_AUDIO_POLL_INTERVAL = 0.002
WEBRTC_AUDIO_OPUS_BITRATE = 64_000
WEBRTC_AUDIO_OPUS_SAMPLE_RATE = 48000
WEBRTC_AUDIO_OPUS_FRAME_SIZE = int(WEBRTC_AUDIO_OPUS_SAMPLE_RATE * WEBRTC_AUDIO_PTIME)
WEBRTC_AUDIO_MAX_LATENESS = WEBRTC_AUDIO_MAX_BUFFER
WEBRTC_AUDIO_AEC_STREAM_DELAY_MS = 50
WEBRTC_AUDIO_AEC_SAMPLE_RATES = {16000, 32000, 48000}
WEBRTC_AUDIO_NS_LEVEL = 2
WEBRTC_AUDIO_AGC_SPEECH_PROBABILITY = 0.80
WEBRTC_AUDIO_AGC_MIN_RMS = 50.0
WEBRTC_AUDIO_FAR_ACTIVE_RMS = 200.0
WEBRTC_AUDIO_RESIDUAL_ECHO_GAIN = 0.15


def audio_bytes_for_duration(duration: float, sample_rate: int, channels: int = WEBRTC_AUDIO_CHANNELS) -> int:
  return int(duration * sample_rate) * channels * WEBRTC_AUDIO_BYTES_PER_SAMPLE


def even_audio_bytes(size: int) -> int:
  return size - (size % WEBRTC_AUDIO_BYTES_PER_SAMPLE)


def audio_rms(samples: np.ndarray) -> float:
  if samples.size == 0:
    return 0.0
  return float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))


def scale_int16_audio(samples: np.ndarray, gain: float) -> np.ndarray:
  return np.clip(samples.astype(np.float32) * gain, -32768, 32767).astype(np.int16)


def configure_opus_encoder_bitrate(bit_rate: int = WEBRTC_AUDIO_OPUS_BITRATE,
                                   frame_size: int = WEBRTC_AUDIO_OPUS_FRAME_SIZE) -> None:
  from aiortc.codecs.opus import OpusEncoder
  from av.audio.resampler import AudioResampler

  original_init = getattr(OpusEncoder, "_openpilot_original_init", None)
  if original_init is None:
    original_init = OpusEncoder.__init__
    OpusEncoder._openpilot_original_init = original_init

  def patched_init(self) -> None:
    original_init(self)
    self.codec.bit_rate = bit_rate
    self.resampler = AudioResampler(
      format="s16",
      layout="stereo",
      rate=WEBRTC_AUDIO_OPUS_SAMPLE_RATE,
      frame_size=frame_size,
    )

  OpusEncoder.__init__ = patched_init
  OpusEncoder._openpilot_bitrate = bit_rate
  OpusEncoder._openpilot_frame_size = frame_size


class WebRTCAudioProcessor:
  def __init__(self, sample_rate: int = micd.SAMPLE_RATE, channels: int = WEBRTC_AUDIO_CHANNELS,
               echo_canceller_cls=None, noise_suppressor_cls=None, voice_detector_cls=None, gain_controller_cls=None):
    self.channels = channels
    self.echo_canceller_cls = echo_canceller_cls
    self.noise_suppressor_cls = noise_suppressor_cls
    self.voice_detector_cls = voice_detector_cls
    self.gain_controller_cls = gain_controller_cls
    self.sample_rate: int | None = None
    self.enabled = False
    self.echo_canceller = None
    self.noise_suppressor = None
    self.voice_detector = None
    self.gain_controller = None
    self.speech_probability = 0.0
    self.gain_db = 0.0
    self._init_processors(sample_rate)

  def _init_processors(self, sample_rate: int) -> None:
    self.sample_rate = sample_rate
    self.enabled = False
    self.echo_canceller = None
    self.noise_suppressor = None
    self.voice_detector = None
    self.gain_controller = None
    self.speech_probability = 0.0
    self.gain_db = 0.0

    if sample_rate not in WEBRTC_AUDIO_AEC_SAMPLE_RATES or self.channels != WEBRTC_AUDIO_CHANNELS:
      print(f"webrtcd audio processing requires {sorted(WEBRTC_AUDIO_AEC_SAMPLE_RATES)} Hz mono audio", flush=True)
      return

    try:
      echo_canceller_cls = self.echo_canceller_cls
      noise_suppressor_cls = self.noise_suppressor_cls
      voice_detector_cls = self.voice_detector_cls
      gain_controller_cls = self.gain_controller_cls
      if echo_canceller_cls is None or noise_suppressor_cls is None or voice_detector_cls is None or gain_controller_cls is None:
        from pywebrtc_audio import EchoCanceller, GainController, NoiseSuppressor, VoiceDetector
        echo_canceller_cls = echo_canceller_cls or EchoCanceller
        noise_suppressor_cls = noise_suppressor_cls or NoiseSuppressor
        voice_detector_cls = voice_detector_cls or VoiceDetector
        gain_controller_cls = gain_controller_cls or GainController

      self.echo_canceller = echo_canceller_cls(
        sample_rate=sample_rate,
        num_channels=self.channels,
        stream_delay_ms=WEBRTC_AUDIO_AEC_STREAM_DELAY_MS,
      )
      self.noise_suppressor = noise_suppressor_cls(
        sample_rate=sample_rate,
        num_channels=self.channels,
        level=WEBRTC_AUDIO_NS_LEVEL,
      )
      self.voice_detector = voice_detector_cls(
        sample_rate=sample_rate,
        num_channels=self.channels,
      )
      self.gain_controller = gain_controller_cls(
        sample_rate=sample_rate,
        num_channels=self.channels,
      )
      self.enabled = True
      print("webrtcd AEC/NS/VAD/AGC enabled for outgoing audio", flush=True)
    except ImportError as e:
      print(f"webrtcd audio processing unavailable: {e}", flush=True)
    except Exception as e:
      print(f"webrtcd failed to initialize audio processing: {e!r}", flush=True)
      self.disable()

  def reset(self) -> None:
    for processor in (self.echo_canceller, self.noise_suppressor, self.voice_detector, self.gain_controller):
      if processor is not None and hasattr(processor, "reset"):
        processor.reset()
    self.speech_probability = 0.0
    self.gain_db = 0.0

  def disable(self) -> None:
    self.enabled = False
    self.echo_canceller = None
    self.noise_suppressor = None
    self.voice_detector = None
    self.gain_controller = None
    self.speech_probability = 0.0
    self.gain_db = 0.0

  def process(self, near_data: bytes, far_data: bytes, sample_rate: int) -> bytes:
    if not near_data:
      return near_data

    if sample_rate != self.sample_rate:
      self._init_processors(sample_rate)

    if not self.enabled or self.echo_canceller is None or self.noise_suppressor is None or self.voice_detector is None or self.gain_controller is None:
      return near_data

    near_bytes = even_audio_bytes(len(near_data))
    if near_bytes != len(near_data):
      return near_data

    if len(far_data) < near_bytes:
      far_data += bytes(near_bytes - len(far_data))
    elif len(far_data) > near_bytes:
      far_data = far_data[:near_bytes]

    try:
      near = np.frombuffer(near_data, dtype=np.int16)
      far = np.frombuffer(far_data, dtype=np.int16)
      echo_cancelled = self.echo_canceller.process(near, far)
      noise_suppressed = self.noise_suppressor.process(echo_cancelled)
      self.speech_probability = float(self.voice_detector.process(noise_suppressed))

      speech_detected = self.speech_probability >= WEBRTC_AUDIO_AGC_SPEECH_PROBABILITY and audio_rms(noise_suppressed) >= WEBRTC_AUDIO_AGC_MIN_RMS
      far_active = audio_rms(far) >= WEBRTC_AUDIO_FAR_ACTIVE_RMS
      if speech_detected:
        processed = self.gain_controller.process(noise_suppressed, self.speech_probability)
        self.gain_db = float(getattr(self.gain_controller, "gain_db", 0.0))
      elif far_active:
        processed = scale_int16_audio(noise_suppressed, WEBRTC_AUDIO_RESIDUAL_ECHO_GAIN)
        self.gain_db = 0.0
      else:
        processed = noise_suppressed
        self.gain_db = 0.0

      data = np.asarray(processed, dtype=np.int16).tobytes()
      if len(data) != len(near_data):
        raise RuntimeError(f"audio processing changed packet size: {len(near_data)} -> {len(data)}")
      return data
    except Exception as e:
      print(f"webrtcd audio processing failed; disabling: {e!r}", flush=True)
      self.disable()
      return near_data


class WebRTCAudioEchoCanceller(WebRTCAudioProcessor):
  pass


class LiveStreamAudioStreamTrack(MediaStreamTrack):
  kind = "audio"

  def __init__(self, echo_canceller: WebRTCAudioProcessor | None = None):
    configure_opus_encoder_bitrate()
    super().__init__()
    self._sock = messaging.sub_sock("rawAudioData", conflate=False)
    self._far_audio_sock = messaging.sub_sock(WEBRTC_AUDIO_SERVICE, conflate=False)
    self._start: float | None = None
    self._timestamp = 0
    self._sample_rate = micd.SAMPLE_RATE
    self._audio_buffer = bytearray()
    self._far_audio_buffer = bytearray()
    self._far_audio_resampler = None
    self._far_audio_resampler_sample_rates: tuple[int, int] | None = None
    self._primed = False
    self._audio_processor = echo_canceller if echo_canceller is not None else WebRTCAudioProcessor()

  def stop(self) -> None:
    super().stop()
    self._sock = None
    self._far_audio_sock = None

  def _trim_audio_buffer(self) -> None:
    max_bytes = audio_bytes_for_duration(WEBRTC_AUDIO_MAX_BUFFER, self._sample_rate)
    if len(self._audio_buffer) <= max_bytes:
      return

    drop_bytes = len(self._audio_buffer) - max_bytes
    drop_bytes = even_audio_bytes(drop_bytes)
    del self._audio_buffer[:drop_bytes]

  def _trim_far_audio_buffer(self) -> None:
    max_bytes = audio_bytes_for_duration(WEBRTC_AUDIO_REFERENCE_MAX_BUFFER, self._sample_rate)
    if len(self._far_audio_buffer) <= max_bytes:
      return

    drop_bytes = even_audio_bytes(len(self._far_audio_buffer) - max_bytes)
    del self._far_audio_buffer[:drop_bytes]

  def _reset_timing(self) -> None:
    self._audio_buffer.clear()
    self._far_audio_buffer.clear()
    self._far_audio_resampler = None
    self._far_audio_resampler_sample_rates = None
    self._start = None
    self._timestamp = 0
    self._primed = False
    if hasattr(self._audio_processor, "reset"):
      self._audio_processor.reset()

  def _target_buffer_bytes(self) -> int:
    return audio_bytes_for_duration(WEBRTC_AUDIO_PTIME + WEBRTC_AUDIO_TARGET_BUFFER, self._sample_rate)

  def _resample_far_audio(self, data: bytes, sample_rate: int) -> bytes:
    if sample_rate == self._sample_rate or not data:
      return data

    data = data[:even_audio_bytes(len(data))]
    if not data:
      return data

    from av.audio.resampler import AudioResampler

    sample_rates = (sample_rate, self._sample_rate)
    if self._far_audio_resampler is None or self._far_audio_resampler_sample_rates != sample_rates:
      self._far_audio_resampler = AudioResampler(format="s16", layout="mono", rate=self._sample_rate)
      self._far_audio_resampler_sample_rates = sample_rates

    frame = AudioFrame(format="s16", layout="mono", samples=len(data) // WEBRTC_AUDIO_BYTES_PER_SAMPLE)
    frame.planes[0].update(data)
    frame.sample_rate = sample_rate

    resampled = bytearray()
    for resampled_frame in self._far_audio_resampler.resample(frame):
      resampled.extend(resampled_frame.to_ndarray().astype(np.int16, copy=False).tobytes())
    return bytes(resampled)

  def _drain_far_audio_messages(self) -> None:
    while True:
      msg = messaging.recv_one_or_none(self._far_audio_sock)
      if msg is None:
        break

      sample_rate = msg.webrtcAudioData.sampleRate
      if sample_rate <= 0:
        continue
      data = self._resample_far_audio(bytes(msg.webrtcAudioData.data), sample_rate)
      self._far_audio_buffer.extend(data)
      self._trim_far_audio_buffer()

  def _read_far_audio_reference(self, target_bytes: int) -> bytes:
    self._drain_far_audio_messages()
    data = bytes(self._far_audio_buffer[:target_bytes])
    del self._far_audio_buffer[:target_bytes]

    if len(data) < target_bytes:
      data += bytes(target_bytes - len(data))

    return data

  def _drain_audio_messages(self) -> None:
    while True:
      msg = messaging.recv_one_or_none(self._sock)
      if msg is None:
        break

      sample_rate = msg.rawAudioData.sampleRate
      if sample_rate != self._sample_rate:
        self._sample_rate = sample_rate
        self._reset_timing()

      self._audio_buffer.extend(bytes(msg.rawAudioData.data))
      self._trim_audio_buffer()

    self._drain_far_audio_messages()

  async def _fill_audio_buffer(self, target_bytes: int, timeout: float = WEBRTC_AUDIO_PTIME) -> None:
    deadline = time.monotonic() + timeout
    while len(self._audio_buffer) < target_bytes:
      if self.readyState != "live":
        raise MediaStreamError

      self._drain_audio_messages()
      if len(self._audio_buffer) >= target_bytes:
        break

      if time.monotonic() >= deadline:
        break

      await asyncio.sleep(WEBRTC_AUDIO_POLL_INTERVAL)

  async def _prime_audio_buffer(self) -> None:
    if self._primed:
      return

    await self._fill_audio_buffer(self._target_buffer_bytes(), WEBRTC_AUDIO_INITIAL_BUFFER)
    self._primed = True

  async def _next_audio_data(self) -> tuple[bytes, int]:
    await self._prime_audio_buffer()

    await self._fill_audio_buffer(self._target_buffer_bytes())
    target_bytes = audio_bytes_for_duration(WEBRTC_AUDIO_PTIME, self._sample_rate)

    if len(self._audio_buffer) >= target_bytes:
      data = bytes(self._audio_buffer[:target_bytes])
      del self._audio_buffer[:target_bytes]
    else:
      data = bytes(self._audio_buffer)
      self._audio_buffer.clear()
      data += bytes(target_bytes - len(data))

    far_data = self._read_far_audio_reference(target_bytes)
    data = self._audio_processor.process(data, far_data, self._sample_rate)

    return data, self._sample_rate

  async def _pace(self, pts: int, sample_rate: int) -> None:
    now = time.monotonic()
    if self._start is None:
      self._start = now
      return

    wait = self._start + (pts / sample_rate) - now
    if wait > 0:
      await asyncio.sleep(wait)
    elif -wait > WEBRTC_AUDIO_MAX_LATENESS:
      self._start = now - (pts / sample_rate)
      self._audio_buffer.clear()
      self._primed = False

  async def recv(self):
    data, sample_rate = await self._next_audio_data()
    samples = len(data) // 2
    pts = self._timestamp
    self._timestamp += samples
    await self._pace(pts, sample_rate)

    frame = AudioFrame(format="s16", layout="mono", samples=samples)
    frame.planes[0].update(data)
    frame.pts = pts
    frame.sample_rate = sample_rate
    frame.time_base = fractions.Fraction(1, sample_rate)
    return frame
