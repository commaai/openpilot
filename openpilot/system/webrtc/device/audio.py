import asyncio
import fractions
import time

from aiortc.mediastreams import MediaStreamError, MediaStreamTrack
from av import AudioFrame

from openpilot.cereal import messaging
from openpilot.system import micd


WEBRTC_AUDIO_SERVICE = "webrtcAudioData"
WEBRTC_AUDIO_PTIME = 0.020


class LiveStreamAudioStreamTrack(MediaStreamTrack):
  kind = "audio"

  def __init__(self):
    super().__init__()
    self._sock = messaging.sub_sock("rawAudioData")
    self._start: float | None = None
    self._timestamp = 0
    self._sample_rate = micd.SAMPLE_RATE
    self._audio_buffer = bytearray()

  def stop(self) -> None:
    super().stop()
    self._sock = None

  async def _fill_audio_buffer(self, target_bytes: int) -> None:
    deadline = time.monotonic() + WEBRTC_AUDIO_PTIME
    while len(self._audio_buffer) < target_bytes:
      if self.readyState != "live":
        raise MediaStreamError

      msg = messaging.recv_one_or_none(self._sock)
      if msg is not None:
        sample_rate = msg.rawAudioData.sampleRate
        if sample_rate != self._sample_rate:
          self._sample_rate = sample_rate
          self._audio_buffer.clear()
          self._start = None
          self._timestamp = 0
        self._audio_buffer.extend(bytes(msg.rawAudioData.data))
        continue

      if time.monotonic() >= deadline:
        break

      await asyncio.sleep(0.005)

  async def _next_audio_data(self) -> tuple[bytes, int]:
    samples = int(WEBRTC_AUDIO_PTIME * self._sample_rate)
    target_bytes = samples * 2
    await self._fill_audio_buffer(target_bytes)

    if len(self._audio_buffer) >= target_bytes:
      data = bytes(self._audio_buffer[:target_bytes])
      del self._audio_buffer[:target_bytes]
    else:
      data = bytes(self._audio_buffer)
      self._audio_buffer.clear()
      data += bytes(target_bytes - len(data))

    return data, self._sample_rate

  async def _pace(self, pts: int, sample_rate: int) -> None:
    if self._start is None:
      self._start = time.monotonic()
      return

    wait = self._start + (pts / sample_rate) - time.monotonic()
    if wait > 0:
      await asyncio.sleep(wait)

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
