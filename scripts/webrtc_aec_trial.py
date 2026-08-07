#!/usr/bin/env python3
"""Measure speaker-to-microphone echo through a two-way WebRTC session.

The client sends a deterministic broadband signal to webrtcd and records the
microphone track returned by the same call. It then estimates the end-to-end
acoustic delay and the level of the correlated echo.
"""

import argparse
import asyncio
import fractions
import json
import time
import urllib.request
import wave
from collections import deque

import av
import numpy as np

from teleoprtc import StreamingOffer, WebRTCOfferBuilder
from teleoprtc.stream import RTCSessionDescription


SAMPLE_RATE = 48000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000


class ProbeAudioTrack:
  kind = "audio"
  id = "aec-probe"

  def __init__(self, duration: float, amplitude: float, seed: int, input_wav: str | None = None,
               wav_start: float = 0.):
    self.readyState = "live"
    self.total_samples = round(duration * SAMPLE_RATE)
    if input_wav is not None:
      with wave.open(input_wav, "rb") as wav:
        if wav.getsampwidth() != 2 or wav.getframerate() != SAMPLE_RATE:
          raise ValueError("input WAV must contain 16-bit 48 kHz PCM")
        wav.setpos(min(round(wav_start * SAMPLE_RATE), wav.getnframes()))
        raw = np.frombuffer(wav.readframes(self.total_samples), dtype=np.int16)
        if wav.getnchannels() > 1:
          raw = raw.reshape(-1, wav.getnchannels()).astype(np.int32).mean(axis=1)
        shaped = np.zeros(self.total_samples, dtype=np.float32)
        shaped[:len(raw)] = raw
    else:
      rng = np.random.default_rng(seed)
      # Band-limit random noise to resemble broadband speech while retaining a
      # sharp, unambiguous cross-correlation peak.
      raw = rng.standard_normal(self.total_samples + 8).astype(np.float32)
      shaped = np.convolve(raw, np.array([.08, .16, .22, .24, .16, .08, .04, .02], dtype=np.float32), mode="valid")
    shaped /= max(float(np.max(np.abs(shaped))), 1e-9)
    self.samples = np.clip(shaped * amplitude * 32767, -32768, 32767).astype(np.int16)
    self.encoder = av.CodecContext.create("opus", "w")
    self.encoder.sample_rate = SAMPLE_RATE
    self.encoder.layout = "mono"
    self.encoder.format = "s16"
    self.encoder.bit_rate = 64_000
    self.encoder.time_base = fractions.Fraction(1, SAMPLE_RATE)
    self.encoder.open()
    self.packets: deque[av.Packet] = deque()
    self.position = 0
    self.next_send_time: float | None = None

  def stop(self) -> None:
    self.readyState = "ended"

  async def recv(self) -> av.Packet:
    if self.packets:
      return self.packets.popleft()
    if self.position >= self.total_samples:
      pcm = np.zeros(FRAME_SAMPLES, dtype=np.int16)
    else:
      pcm = np.zeros(FRAME_SAMPLES, dtype=np.int16)
      count = min(FRAME_SAMPLES, self.total_samples - self.position)
      pcm[:count] = self.samples[self.position:self.position + count]
      self.position += count

    frame = av.AudioFrame.from_ndarray(pcm.reshape(1, -1), format="s16", layout="mono")
    frame.sample_rate = SAMPLE_RATE
    frame.pts = self.position
    frame.time_base = fractions.Fraction(1, SAMPLE_RATE)
    for packet in self.encoder.encode(frame):
      packet.time_base = fractions.Fraction(1, SAMPLE_RATE)
      self.packets.append(packet)

    now = time.monotonic()
    if self.next_send_time is None or now - self.next_send_time > FRAME_MS / 1000:
      self.next_send_time = now
    await asyncio.sleep(max(0., self.next_send_time - now))
    self.next_send_time += FRAME_MS / 1000
    return self.packets.popleft()


class ConnectionProvider:
  def __init__(self, host: str, port: int, timeout: float):
    self.url = f"http://{host}:{port}/stream"
    self.timeout = timeout

  async def __call__(self, offer: StreamingOffer) -> RTCSessionDescription:
    return await asyncio.to_thread(self._post, offer)

  def _post(self, offer: StreamingOffer) -> RTCSessionDescription:
    body = {"sdp": offer.sdp, "cameras": [], "enabled": True,
            "bridge_services_in": [], "bridge_services_out": []}
    request = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=self.timeout) as response:
      payload = json.loads(response.read())
    if "error" in payload:
      raise RuntimeError(payload.get("message", payload["error"]))
    return RTCSessionDescription(sdp=payload["sdp"], type=payload["type"])


async def receive_audio(track, duration: float) -> np.ndarray:
  decoder = av.CodecContext.create("opus", "r")
  resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
  chunks: list[np.ndarray] = []
  deadline = time.monotonic() + duration
  while time.monotonic() < deadline:
    packet = await asyncio.to_thread(track.receive)
    if packet is None:
      await asyncio.sleep(.002)
      continue
    for decoded in decoder.decode(av.Packet(bytes(packet))):
      frames = resampler.resample(decoded)
      for frame in frames if isinstance(frames, list) else [frames]:
        if frame is not None:
          chunks.append(frame.to_ndarray().reshape(-1).copy())
  return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int16)


def measure(reference: np.ndarray, capture: np.ndarray, max_delay_ms: int) -> dict[str, float]:
  # Skip call startup and AEC adaptation, then locate the probe in the returned
  # microphone audio. FFT correlation keeps multi-second trials inexpensive.
  skip = SAMPLE_RATE * 2
  reference = reference[skip:].astype(np.float64)
  capture = capture.astype(np.float64)
  reference -= np.mean(reference)
  capture -= np.mean(capture) if len(capture) else 0
  n = 1 << (len(reference) + len(capture) - 1).bit_length()
  correlation = np.fft.irfft(np.fft.rfft(capture, n) * np.conj(np.fft.rfft(reference, n)), n)
  max_lag = min(len(capture) - 1, max_delay_ms * SAMPLE_RATE // 1000 + SAMPLE_RATE * 3)
  lag = int(np.argmax(np.abs(correlation[:max_lag + 1])))
  count = min(len(reference), len(capture) - lag)
  if count <= 0:
    raise RuntimeError("not enough returned microphone audio to measure echo")
  ref = reference[:count]
  mic = capture[lag:lag + count]
  gain = float(np.dot(ref, mic) / max(np.dot(ref, ref), 1e-12))
  correlated = ref * gain
  ref_rms = float(np.sqrt(np.mean(ref * ref)))
  echo_rms = float(np.sqrt(np.mean(correlated * correlated)))
  mic_rms = float(np.sqrt(np.mean(mic * mic)))
  return {
    "delay_ms": lag * 1000 / SAMPLE_RATE,
    "reference_dbfs": 20 * np.log10(max(ref_rms / 32768, 1e-12)),
    "correlated_echo_dbfs": 20 * np.log10(max(echo_rms / 32768, 1e-12)),
    "microphone_dbfs": 20 * np.log10(max(mic_rms / 32768, 1e-12)),
    "echo_gain_db": 20 * np.log10(max(abs(gain), 1e-12)),
  }


async def run(args: argparse.Namespace) -> None:
  probe = ProbeAudioTrack(args.duration, args.amplitude, args.seed, args.input_wav, args.wav_start)
  builder = WebRTCOfferBuilder(ConnectionProvider(args.host, args.port, args.timeout))
  builder.add_audio_stream(probe)
  if not args.send_only:
    builder.offer_to_receive_audio_stream()
  stream = builder.stream()
  try:
    await asyncio.wait_for(stream.start(), args.timeout)
    await asyncio.wait_for(stream.wait_for_connection(), args.timeout)
    if args.send_only:
      await asyncio.sleep(args.duration + args.tail)
      print(json.dumps({"sent_samples": probe.position}, indent=2))
      return
    capture = await receive_audio(stream.get_incoming_audio_track(), args.duration + args.tail)
  finally:
    await stream.stop()
  result = measure(probe.samples, capture, args.max_delay_ms)
  result.update(capture_samples=len(capture), probe_samples=len(probe.samples))
  print(json.dumps(result, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--host", required=True)
  parser.add_argument("--port", type=int, default=5001)
  parser.add_argument("--duration", type=float, default=12.)
  parser.add_argument("--tail", type=float, default=1.)
  parser.add_argument("--amplitude", type=float, default=.25)
  parser.add_argument("--seed", type=int, default=20260805)
  parser.add_argument("--input-wav", help="16-bit, 48 kHz WAV to use instead of the generated probe")
  parser.add_argument("--wav-start", type=float, default=0., help="Starting offset within --input-wav")
  parser.add_argument("--max-delay-ms", type=int, default=1000)
  parser.add_argument("--timeout", type=float, default=15.)
  parser.add_argument("--send-only", action="store_true", help="Send the probe without negotiating returned microphone audio")
  return parser.parse_args()


if __name__ == "__main__":
  asyncio.run(run(parse_args()))
