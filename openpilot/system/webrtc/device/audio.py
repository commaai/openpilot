"""On-demand, full-duplex Opus audio for livestream sessions."""
import asyncio
import contextlib
import ctypes
import ctypes.util
import logging
import queue
import random
import time

import numpy as np

from openpilot.cereal import messaging
from openpilot.system.micd import SAMPLE_RATE as MIC_RATE

from libdatachannel import (Description, FrameInfo, OpusRtpPacketizer, OpusRtpDepacketizer,
                            RtcpReceivingSession, RtcpSrReporter, RtpPacketizationConfig)
from teleoprtc.builder import WebRTCAnswerBuilder
from teleoprtc.stream import RTCSessionDescription, WebRTCAnswerStream

RATE = 48000
SAMPLES = 960  # 20 ms, mono signed 16-bit PCM
MICROPHONE_GAIN = 8.0  # +18 dB for device audio heard in Connect


class OpusCodec:
  def __init__(self, encoder_rate=RATE):
    self.encoder_samples = encoder_rate // 50
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
    self.encoder = self.lib.opus_encoder_create(encoder_rate, 1, 2048, ctypes.byref(error))  # OPUS_APPLICATION_VOIP
    if error.value or not self.encoder:
      self.close()
      raise RuntimeError(f"Opus encoder: {error.value}")
    self.decoder = self.lib.opus_decoder_create(RATE, 1, ctypes.byref(error))
    if error.value or not self.decoder:
      self.close()
      raise RuntimeError(f"Opus decoder: {error.value}")

  def encode(self, pcm: bytes) -> bytes:
    if len(pcm) != self.encoder_samples * 2:
      raise ValueError("Expected 20 ms of mono PCM")
    output = ctypes.create_string_buffer(4000)
    size = self.lib.opus_encode(self.encoder, pcm, self.encoder_samples, output, len(output))
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
    self.received = queue.Queue(maxsize=6)
    self.task = None
    self.closed = False
    self.timestamp_base = random.randint(0, 0xFFFFFFFF)
    self.capture_pending = bytearray()
    self.capture_time = None
    self.last_capture_time = None
    self.last_received_timestamp = None
    track.on_frame(self.on_frame)

  def on_frame(self, packet, info):
    if self.closed or len(packet) > 4000:
      return
    timestamp = info.timestamp
    if self.last_received_timestamp is not None:
      delta = (timestamp - self.last_received_timestamp) & 0xFFFFFFFF
      if delta == 0 or delta >= 0x80000000:
        return  # Do not replay duplicate or out-of-order voice packets.
    self.last_received_timestamp = timestamp
    put_latest(self.received, (time.monotonic_ns(), bytes(packet)))

  def enable(self, enabled):
    self.enabled = bool(enabled)
    if not self.enabled:
      self.capture_pending.clear()
      self.capture_time = None
      self.last_capture_time = None

  def start(self):
    if self.task is None:
      self.task = asyncio.create_task(self.run())

  def send_capture(self, msg, codec):
    if not self.enabled or not self.track.is_open():
      return
    audio = msg.rawAudioData
    if audio.sampleRate != MIC_RATE or not audio.data or len(audio.data) % 2:
      return
    if time.monotonic_ns() - msg.logMonoTime > 200_000_000:
      return
    # rawAudioData contains 50 ms at 16 kHz; Opus uses 20 ms frames.
    # Anchor timestamps to capture time so mute/unmute and dropped samples
    # preserve the RTP timeline instead of slowing the receiver's clock.
    start = msg.logMonoTime / 1e9 - len(audio.data) / (MIC_RATE * 2)
    if self.capture_time is None or self.last_capture_time is None or abs(start - self.last_capture_time) > 0.02:
      self.capture_pending.clear()
      self.capture_time = start
    self.last_capture_time = start + len(audio.data) / (MIC_RATE * 2)
    self.capture_pending.extend(audio.data)
    size = codec.encoder_samples * 2
    while len(self.capture_pending) >= size:
      pcm = bytes(self.capture_pending[:size])
      del self.capture_pending[:size]
      timestamp = (self.timestamp_base + round(self.capture_time * RATE)) & 0xFFFFFFFF
      # Boost only the livestream copy, leaving micd's ambient measurement unchanged.
      # Soft limiting keeps loud peaks in range without int16 wraparound.
      samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768
      pcm = (np.tanh(samples * MICROPHONE_GAIN) * 32767).astype(np.int16).tobytes()
      self.track.send_frame(codec.encode(pcm), FrameInfo(timestamp))
      self.capture_time += 0.02

  async def run(self):
    codec = None
    microphone = None
    speaker = None
    try:
      codec = OpusCodec(encoder_rate=MIC_RATE)
      microphone = messaging.sub_sock("rawAudioData")
      speaker = messaging.PubMaster(["livestreamAudio"])
      while not self.closed:
        for _ in range(8):
          msg = messaging.recv_one_or_none(microphone)
          if msg is None:
            break
          self.send_capture(msg, codec)
        for _ in range(6):
          try:
            received_at, packet = self.received.get_nowait()
          except queue.Empty:
            break
          if time.monotonic_ns() - received_at > 200_000_000:
            continue
          try:
            pcm = codec.decode(packet)
          except ValueError:
            continue
          msg = messaging.new_message("livestreamAudio", valid=True)
          msg.livestreamAudio.sampleRate = RATE
          msg.livestreamAudio.data = pcm
          speaker.send("livestreamAudio", msg)
        await asyncio.sleep(0.005)
    except Exception:
      logging.getLogger("webrtcd").exception("Livestream audio failed")
      self.on_error()
    finally:
      if speaker is not None:
        # Flush soundd's short playback queue on disconnect.
        msg = messaging.new_message("livestreamAudio", valid=True)
        msg.livestreamAudio.sampleRate = RATE
        speaker.send("livestreamAudio", msg)
      if codec is not None:
        codec.close()

  async def stop(self):
    self.closed = True
    self.enable(False)
    if self.task is not None:
      await asyncio.shield(self.task)
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
