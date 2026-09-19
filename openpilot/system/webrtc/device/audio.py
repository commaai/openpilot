"""On-demand, full-duplex Opus audio for livestream sessions."""
import asyncio
import contextlib
import ctypes
import ctypes.util
import logging
import queue
import random

from libdatachannel import (Description, FrameInfo, OpusRtpPacketizer, OpusRtpDepacketizer,
                            RtcpReceivingSession, RtcpSrReporter, RtpPacketizationConfig)
from teleoprtc.builder import WebRTCAnswerBuilder
from teleoprtc.stream import RTCSessionDescription, WebRTCAnswerStream

RATE = 48000
SAMPLES = 960  # 20 ms, mono signed 16-bit PCM


class OpusCodec:
  def __init__(self):
    self.lib = ctypes.CDLL(ctypes.util.find_library("opus") or "libopus.so.0")
    pointer = ctypes.c_void_p
    integer = ctypes.c_int
    for name, args, result in (
      ("opus_encoder_create", [integer, integer, integer, ctypes.POINTER(integer)], pointer),
      ("opus_decoder_create", [integer, integer, ctypes.POINTER(integer)], pointer),
      ("opus_encode", [pointer, pointer, integer, pointer, integer], integer),
      ("opus_decode", [pointer, pointer, integer, pointer, integer, integer], integer),
      ("opus_encoder_destroy", [pointer], None),
      ("opus_decoder_destroy", [pointer], None),
    ):
      fn = getattr(self.lib, name)
      fn.argtypes, fn.restype = args, result
    self.encoder = None
    self.decoder = None
    error = integer()
    self.encoder = self.lib.opus_encoder_create(RATE, 1, 2048, ctypes.byref(error))  # OPUS_APPLICATION_VOIP
    if error.value or not self.encoder:
      self.close()
      raise RuntimeError(f"Opus encoder: {error.value}")
    self.decoder = self.lib.opus_decoder_create(RATE, 1, ctypes.byref(error))
    if error.value or not self.decoder:
      self.close()
      raise RuntimeError(f"Opus decoder: {error.value}")

  def encode(self, pcm: bytes) -> bytes:
    if len(pcm) != SAMPLES * 2:
      raise ValueError("Expected 20 ms of mono PCM")
    output = ctypes.create_string_buffer(4000)
    size = self.lib.opus_encode(self.encoder, pcm, SAMPLES, output, len(output))
    if size < 0:
      raise ValueError(f"Opus encode: {size}")
    return output.raw[:size]

  def decode(self, packet: bytes) -> bytes:
    output = ctypes.create_string_buffer(5760 * 2)  # maximum Opus packet duration: 120 ms
    size = self.lib.opus_decode(self.decoder, packet, len(packet), output, 5760, 0)
    if size < 0:
      raise ValueError(f"Opus decode: {size}")
    return output.raw[:size * 2]

  def close(self):
    if self.encoder:
      self.lib.opus_encoder_destroy(self.encoder)
      self.encoder = None
    if self.decoder:
      self.lib.opus_decoder_destroy(self.decoder)
      self.decoder = None


def put_latest(buffer: queue.Queue, value):
  try:
    buffer.put_nowait(value)
  except queue.Full:
    with contextlib.suppress(queue.Empty):
      buffer.get_nowait()
    with contextlib.suppress(queue.Full):
      buffer.put_nowait(value)


class LivestreamAudio:
  def __init__(self, track, on_error):
    self.track = track
    self.on_error = on_error
    self.enabled = False
    self.capture = queue.Queue(maxsize=3)
    self.received = queue.Queue(maxsize=6)
    self.playback = queue.Queue(maxsize=6)
    self.task = None
    self.closed = False
    self.timestamp = random.randint(0, 0xFFFFFFFF)
    track.on_frame(self.on_frame)

  def on_frame(self, packet, _info):
    if not self.closed and len(packet) <= 4000:
      put_latest(self.received, bytes(packet))

  def enable(self, enabled):
    self.enabled = bool(enabled)

  def start(self):
    self.task = asyncio.create_task(self.run())

  async def run(self):
    import sounddevice as sd
    codec = None
    mic = None
    speaker = None
    pending = bytearray()

    def capture(data, frames, _time, _status):
      if self.enabled and frames == SAMPLES:
        put_latest(self.capture, bytes(data))

    def playback(data, _frames, _time, _status):
      while len(pending) < len(data):
        try:
          pending.extend(self.playback.get_nowait())
        except queue.Empty:
          break
      count = min(len(data), len(pending))
      data[:count] = bytes(pending[:count])
      data[count:] = bytes(len(data) - count)
      del pending[:count]

    try:
      codec = OpusCodec()
      # PortAudio opens lazily: prewarm does not capture or play any audio.
      while not self.closed:
        if self.enabled and mic is None:
          mic = await asyncio.to_thread(sd.RawInputStream, samplerate=RATE, channels=1, dtype="int16", blocksize=SAMPLES, callback=capture)
          await asyncio.to_thread(mic.start)
        elif not self.enabled and mic is not None:
          await asyncio.to_thread(mic.close)
          mic = None
          while not self.capture.empty():
            self.capture.get_nowait()
        for _ in range(3):
          try:
            pcm = self.capture.get_nowait()
          except queue.Empty:
            break
          if self.enabled and self.track.is_open():
            self.track.send_frame(codec.encode(pcm), FrameInfo(self.timestamp))
            self.timestamp = (self.timestamp + SAMPLES) & 0xFFFFFFFF
        for _ in range(6):
          try:
            packet = self.received.get_nowait()
          except queue.Empty:
            break
          try:
            pcm = codec.decode(packet)
          except ValueError:
            continue
          if speaker is None:
            speaker = await asyncio.to_thread(sd.RawOutputStream, samplerate=RATE, channels=1, dtype="int16", blocksize=SAMPLES, callback=playback)
            await asyncio.to_thread(speaker.start)
          put_latest(self.playback, pcm)
        await asyncio.sleep(0.005)
    except Exception:
      logging.getLogger("webrtcd").exception("Livestream audio failed")
      self.on_error()
    finally:
      for device in (mic, speaker):
        if device is not None:
          await asyncio.to_thread(device.close)
      if codec is not None:
        codec.close()

  async def stop(self):
    self.closed = True
    self.enabled = False
    # Let in-flight PortAudio opens finish before closing their handles.
    if self.task is not None:
      await self.task
      self.task = None


class AudioAnswerStream(WebRTCAnswerStream):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.audio = None

  def _add_producer_tracks(self, remote_sdp=None):
    super()._add_producer_tracks(remote_sdp)
    offer = Description(remote_sdp, Description.Type.Offer)
    for i in range(offer.media_count()):
      media = offer.media(i)
      if media is None or media.type() != "audio" or media.direction() != Description.Direction.SendRecv:
        continue
      for payload in media.payload_types():
        mapping = media.rtp_map(payload)
        if mapping is None or mapping.format.lower() != "opus":
          continue
        ssrc = random.randint(1, 0xFFFFFFFF)
        answer = Description.Audio(media.mid(), Description.Direction.SendRecv)
        answer.add_opus_codec(payload)
        answer.add_ssrc(ssrc, "livestream-audio", "audio", "microphone")
        track = self.peer_connection.add_track(answer)
        config = RtpPacketizationConfig(ssrc, "livestream-audio", payload, RATE)
        packetizer = OpusRtpPacketizer(config)
        packetizer.add_to_chain(RtcpSrReporter(config))
        packetizer.add_to_chain(OpusRtpDepacketizer())
        packetizer.add_to_chain(RtcpReceivingSession())
        track.set_media_handler(packetizer)
        self.audio = LivestreamAudio(track, self._audio_error)
        # The answer's bidirectional track is also the incoming audio track.
        self.incoming_audio_tracks.append(track)
        self._on_after_media()
        return

  def _audio_error(self):
    if self.messaging_channel is not None and self.messaging_channel.is_open():
      self.messaging_channel.send('{"type":"audioError","data":"Device audio failed. Reconnect to retry."}')

  async def wait_for_connection(self):
    await super().wait_for_connection()
    if self.audio is not None:
      self.audio.start()

  async def stop(self):
    if self.audio is not None:
      await self.audio.stop()
    # Closing native callbacks may wait for their GIL; keep the event loop free.
    await asyncio.to_thread(self.peer_connection.close)
    await super().stop()


class AudioAnswerBuilder(WebRTCAnswerBuilder):
  def stream(self):
    return AudioAnswerStream(
      RTCSessionDescription(sdp=self.offer_sdp, type="offer"),
      consumed_camera_types=[], consume_audio=False,
      video_producer_tracks=list(self.video_tracks.values()), audio_producer_tracks=[],
      should_add_data_channel=False, bind_address=self.bind_address,
    )
