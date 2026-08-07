import asyncio
from collections import deque
import fractions
import logging
import math
import os
import threading
import time

import av
import numpy as np
from pywebrtc_audio import AudioProcessor

from openpilot.cereal import messaging


CAPTURE_RATE = 16000
OPUS_RATE = 48000
FRAME_MS = 10
CAPTURE_FRAME_SAMPLES = CAPTURE_RATE * FRAME_MS // 1000
OPUS_FRAME_SAMPLES = OPUS_RATE * FRAME_MS // 1000
# Keep enough capture for one scheduling hiccup, but never pair a long block of
# pre-connection microphone history with current far-end audio.
SYNC_BUFFER_FRAMES = 5
MAX_CAPTURE_SAMPLES = CAPTURE_FRAME_SAMPLES * (SYNC_BUFFER_FRAMES + 2)
MAX_REFERENCE_SAMPLES = CAPTURE_RATE // 5
SPEECH_PROBABILITY_THRESHOLD = 0.1
SPEECH_HANGOVER_MS = 300


def _env_bool(name: str, default: bool) -> bool:
  value = os.getenv(name)
  return default if value is None else value.lower() not in ("0", "false", "no")


def apply_output_gain(samples: np.ndarray, gain_db: float) -> np.ndarray:
  """Apply fixed makeup gain without wrapping signed 16-bit PCM."""
  if gain_db == 0.:
    return samples
  scale = 10. ** (gain_db / 20.)
  scaled = np.asarray(samples, dtype=np.float64) * scale
  amplified = np.copysign(np.floor(np.abs(scaled) + 0.5), scaled)
  return np.clip(amplified, np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(np.int16)


class SpeechProbabilityGate:
  """Mute non-speech while retaining short pauses between words."""
  def __init__(self, threshold: float, hangover_ms: int):
    if not 0. <= threshold <= 1.:
      raise ValueError("speech probability threshold must be between 0 and 1")
    if hangover_ms < 0:
      raise ValueError("speech hangover must be non-negative")
    self.threshold = threshold
    self.hangover_frames = math.ceil(hangover_ms / FRAME_MS)
    self.remaining_frames = 0

  def reset(self) -> None:
    self.remaining_frames = 0

  def process(self, samples: np.ndarray, speech_probability: float) -> np.ndarray:
    if speech_probability >= self.threshold:
      self.remaining_frames = self.hangover_frames
    elif self.remaining_frames > 0:
      self.remaining_frames -= 1
    else:
      return np.zeros_like(samples)
    return samples


class FarEndAudioBuffer:
  def __init__(self):
    self.samples: deque[int] = deque()
    self.lock = threading.Lock()
    self.total_samples = 0
    self.nonzero_samples = 0

  def add_48khz(self, samples: np.ndarray) -> None:
    downsampled = np.asarray(samples, dtype=np.int16).reshape(-1)[::3]
    with self.lock:
      self.total_samples += len(downsampled)
      self.nonzero_samples += int(np.count_nonzero(downsampled))
      self.samples.extend(int(x) for x in downsampled)
      while len(self.samples) > MAX_REFERENCE_SAMPLES:
        self.samples.popleft()

  def take(self, count: int) -> np.ndarray:
    ret = np.zeros(count, dtype=np.int16)
    with self.lock:
      available = min(count, len(self.samples))
      for i in range(available):
        ret[i] = self.samples.popleft()
    return ret

  def available(self) -> int:
    with self.lock:
      return len(self.samples)

  def clear(self) -> None:
    with self.lock:
      self.samples.clear()

class LiveStreamAudioTrack:
  kind = "audio"

  def __init__(self, far_end: FarEndAudioBuffer):
    self.id = "microphone"
    self.readyState = "live"
    self.far_end = far_end
    self.sock = messaging.sub_sock("rawAudioData", conflate=False)
    self.reference_sock = messaging.sub_sock("webRtcAudioReference", conflate=False)
    self.capture: deque[int] = deque()
    self.packets: deque[av.Packet] = deque()
    self.next_pts = 0
    self.next_packet_pts = 0
    self.next_send_time: float | None = None
    self.synchronized = False
    self.logger = logging.getLogger("webrtcd")
    # Environment overrides make the complete pywebrtc-audio surface available
    # for repeatable hardware sweeps without changing the WebRTC protocol.
    self.processor = AudioProcessor(
      sample_rate=CAPTURE_RATE,
      echo_cancellation=_env_bool("WEBRTC_AEC", True),
      noise_suppression=_env_bool("WEBRTC_NS", True),
      high_pass_filter=_env_bool("WEBRTC_HPF", True),
      auto_gain_control=_env_bool("WEBRTC_AGC", True),
      ns_level=int(os.getenv("WEBRTC_NS_LEVEL", "2")),
      agc_gain_db=float(os.getenv("WEBRTC_AGC_GAIN_DB", "16")),
      agc_max_gain_db=float(os.getenv("WEBRTC_AGC_MAX_GAIN_DB", "6")),
      stream_delay_ms=int(os.getenv("WEBRTC_STREAM_DELAY_MS", "0")),
    )
    self.output_gain_db = float(os.getenv("WEBRTC_OUTPUT_GAIN_DB", "4"))
    self.speech_gate = SpeechProbabilityGate(
      threshold=float(os.getenv("WEBRTC_SPEECH_THRESHOLD", str(SPEECH_PROBABILITY_THRESHOLD))),
      hangover_ms=int(os.getenv("WEBRTC_SPEECH_HANGOVER_MS", str(SPEECH_HANGOVER_MS))),
    )
    self.speech_gate_enabled = _env_bool("WEBRTC_SPEECH_GATE", False)
    self.encoder = av.CodecContext.create("opus", "w")
    self.encoder.sample_rate = OPUS_RATE
    self.encoder.layout = "mono"
    self.encoder.format = "s16"
    self.encoder.bit_rate = 64_000
    self.encoder.time_base = fractions.Fraction(1, OPUS_RATE)
    self.encoder.open()

  def stop(self) -> None:
    self.logger.info("WebRTC audio reference stats: samples=%d nonzero=%d", self.far_end.total_samples,
                     self.far_end.nonzero_samples)
    self.readyState = "ended"

  def _drain_capture(self) -> None:
    for msg in messaging.drain_sock(self.sock):
      if msg.rawAudioData.sampleRate != CAPTURE_RATE:
        self.logger.warning("ignoring rawAudioData at %d Hz", msg.rawAudioData.sampleRate)
        continue
      self.capture.extend(np.frombuffer(msg.rawAudioData.data, dtype=np.int16))
    dropped = len(self.capture) - MAX_CAPTURE_SAMPLES
    for _ in range(max(0, dropped)):
      self.capture.popleft()
    if dropped > 0:
      self.logger.warning("dropped %d stale microphone samples", dropped)

    for msg in messaging.drain_sock(self.reference_sock):
      if msg.webRtcAudioReference.sampleRate != OPUS_RATE:
        self.logger.warning("ignoring webRtcAudioReference at %d Hz", msg.webRtcAudioReference.sampleRate)
        continue
      self.far_end.add_48khz(np.frombuffer(msg.webRtcAudioReference.data, dtype=np.int16))

  def _encode_frame(self) -> None:
    near = np.fromiter((self.capture.popleft() for _ in range(CAPTURE_FRAME_SAMPLES)), dtype=np.int16)
    far = self.far_end.take(CAPTURE_FRAME_SAMPLES)
    try:
      clean = self.processor.process(near, far)
      if self.speech_gate_enabled:
        clean = self.speech_gate.process(clean, self.processor.speech_probability)
    except Exception:
      self.logger.exception("WebRTC audio DSP failed; using raw microphone audio")
      clean = near
    clean = apply_output_gain(clean, self.output_gain_db)

    # 16 kHz and 48 kHz are an integer ratio. Linear interpolation gives an
    # exact 10 ms frame while avoiding codec-resampler startup latency.
    positions = np.arange(OPUS_FRAME_SAMPLES) / 3.
    pcm = np.interp(positions, np.arange(CAPTURE_FRAME_SAMPLES), clean).astype(np.int16)
    frame = av.AudioFrame.from_ndarray(pcm.reshape(1, -1), format="s16", layout="mono")
    frame.sample_rate = OPUS_RATE
    frame.pts = self.next_pts
    frame.time_base = fractions.Fraction(1, OPUS_RATE)
    for packet in self.encoder.encode(frame):
      packet.pts = self.next_packet_pts
      packet.dts = packet.pts
      packet.time_base = fractions.Fraction(1, OPUS_RATE)
      self.packets.append(packet)
      self.next_packet_pts += packet.duration
    self.next_pts += OPUS_FRAME_SAMPLES

  async def recv(self) -> av.Packet:
    while self.readyState == "live":
      self._drain_capture()
      if not self.synchronized:
        # Both subscriptions collect data during SDP/ICE setup. Start their
        # timelines together, then retain a short common backlog so the
        # callback-originated speaker reference has time to cross msgq.
        self.capture.clear()
        self.far_end.clear()
        self.processor.reset()
        self.speech_gate.reset()
        self.synchronized = True
      ready = CAPTURE_FRAME_SAMPLES * SYNC_BUFFER_FRAMES
      if len(self.capture) >= ready and self.far_end.available() >= ready:
        self._encode_frame()
        if self.packets:
          now = time.monotonic()
          if self.next_send_time is None or now - self.next_send_time > FRAME_MS / 1000:
            self.next_send_time = now
          await asyncio.sleep(max(0., self.next_send_time - now))
          self.next_send_time += FRAME_MS / 1000
          return self.packets.popleft()
      await asyncio.sleep(0.005)
    raise asyncio.CancelledError
