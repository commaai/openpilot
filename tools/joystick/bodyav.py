import time
import asyncio
import io
import os
import numpy as np
import pyaudio
import wave

from aiortc.contrib.media import MediaBlackhole
from aiortc.mediastreams import AudioStreamTrack, MediaStreamError
from aiortc.rtcrtpsender import RTCRtpSender
from av import CodecContext, Packet
from pydub import AudioSegment


IMG_H_ORIG, IMG_W_ORIG = 604*2, 964*2
DOWNSCALE = 4
IMG_H = IMG_H_ORIG // DOWNSCALE
IMG_W = IMG_W_ORIG // DOWNSCALE
yuv = np.zeros((int(IMG_H * 1.5 *IMG_W)), dtype=np.uint8)
AUDIO_RATE = 16000
SOUNDS = {
  'engage': '../../selfdrive/assets/sounds/engage.wav',
  'disengage': '../../selfdrive/assets/sounds/disengage.wav',
  'error': '../../selfdrive/assets/sounds/warning_immediate.wav',
}

def force_codec(pc, sender, forced_codec='video/VP9', stream_type="video"):
  codecs = RTCRtpSender.getCapabilities(stream_type).codecs
  codec = [codec for codec in codecs if codec.mimeType == forced_codec]
  transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
  print("transceiver", transceiver, codec)
  transceiver.setCodecPreferences(codec)


import fractions
import time
from typing import Tuple
import cereal.messaging as messaging
from aiortc.mediastreams import (
    VIDEO_CLOCK_RATE,
    VIDEO_PTIME,
    VIDEO_TIME_BASE,
    AudioStreamTrack,
    MediaStreamTrack,
    VideoStreamTrack,
)

class EncodedVideoStream(MediaStreamTrack):
  kind = "video"

  _start: float
  _timestamp: int

  def __init__(self):
    super().__init__()
    sock_name = 'driverEncodeData'
    messaging.context = messaging.Context()
    self.sock = messaging.sub_sock(sock_name, None, conflate=True)

  # TODO I have no idea what this does and if it's right
  async def next_timestamp(self) -> Tuple[int, fractions.Fraction]:
        if hasattr(self, "_timestamp"):
            self._timestamp += int((1/20) * VIDEO_CLOCK_RATE)
            wait = self._start + (self._timestamp / VIDEO_CLOCK_RATE) - time.time()
            await asyncio.sleep(wait)
        else:
            self._start = time.time()
            self._timestamp = 0
        return self._timestamp, VIDEO_TIME_BASE

  async def recv(self) -> Packet:
    pts, time_base = await self.next_timestamp()
    msg = messaging.recv_one(self.sock)
    evta = getattr(msg, msg.which())
    packet = Packet(len(evta.header + evta.data))
    packet.update(bytes(evta.header + evta.data))

    packet.time_base = time_base
    packet.pts = pts
    return packet


class WebClientSpeaker(MediaBlackhole):
  def __init__(self):
    super().__init__()
    self.p = pyaudio.PyAudio()
    self.buffer = io.BytesIO()
    self.channels = 2
    self.stream = self.p.open(format=pyaudio.paInt16, channels=self.channels, rate=48000, frames_per_buffer=9600, output=True, stream_callback=self.pyaudio_callback)

  def pyaudio_callback(self, in_data, frame_count, time_info, status):
    # print(self.buffer.getbuffer().nbytes)
    if self.buffer.getbuffer().nbytes < frame_count * self.channels * 2:
      buff = np.zeros((frame_count, 2), dtype=np.int16).tobytes()
    elif self.buffer.getbuffer().nbytes > 115200:  # 3x the usual read size
      self.buffer.seek(0)
      buff = self.buffer.read(frame_count * self.channels * 4)
      buff = buff[:frame_count * self.channels * 2]
      self.buffer.seek(2)
    else:
      self.buffer.seek(0)
      buff = self.buffer.read(frame_count * self.channels * 2)
      self.buffer.seek(2)
    return (buff, pyaudio.paContinue)

  async def consume(self, track):
    while True:
      try:
        frame = await track.recv()
      except MediaStreamError:
        return
      bio = bytes(frame.planes[0])
      self.buffer.write(bio)

  async def start(self):
    for track, task in self._MediaBlackhole__tracks.items():
      if task is None:
        self._MediaBlackhole__tracks[track] = asyncio.ensure_future(self.consume(track))

  async def stop(self):
    for task in self._MediaBlackhole__tracks.values():
      if task is not None:
        task.cancel()
    self._MediaBlackhole__tracks = {}
    self.stream.stop_stream()
    self.stream.close()
    self.p.terminate()


class BodyMic(AudioStreamTrack):
  def __init__(self):
    super().__init__()

    self.sample_rate = AUDIO_RATE
    self.AUDIO_PTIME = 0.020  # 20ms audio packetization
    self.samples = int(self.AUDIO_PTIME * self.sample_rate)
    self.FORMAT = pyaudio.paInt16
    self.CHANNELS = 2
    self.RATE = self.sample_rate
    self.CHUNK = int(AUDIO_RATE * 0.020)
    self.p = pyaudio.PyAudio()
    self.mic_stream = self.p.open(format=self.FORMAT, channels=1, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)

    self.codec = CodecContext.create('pcm_s16le', 'r')
    self.codec.sample_rate = self.RATE
    self.codec.channels = 2
    self.audio_samples = 0
    self.chunk_number = 0

  async def recv(self):
    mic_data = self.mic_stream.read(self.CHUNK)
    mic_sound = AudioSegment(mic_data, sample_width=2, channels=1, frame_rate=self.RATE)
    mic_sound = AudioSegment.from_mono_audiosegments(mic_sound, mic_sound)
    mic_sound += 3  # increase volume by 3db
    packet = Packet(mic_sound.raw_data)
    frame = self.codec.decode(packet)[0]
    frame.pts = self.audio_samples
    self.audio_samples += frame.samples
    self.chunk_number = self.chunk_number + 1
    return frame


async def play_sound(sound):
  chunk = 5120
  print("playing", sound)
  with wave.open(SOUNDS[sound], 'rb') as wf:
    def callback(in_data, frame_count, time_info, status):
      data = wf.readframes(frame_count)
      return data, pyaudio.paContinue

    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True,
                    frames_per_buffer=chunk,
                    stream_callback=callback)
    stream.start_stream()
    while stream.is_active():
      await asyncio.sleep(0)
    stream.stop_stream()
    stream.close()
    p.terminate()
