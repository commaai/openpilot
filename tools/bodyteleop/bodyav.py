import asyncio
import io
import numpy as np
import pyaudio
import wave

from aiortc.contrib.media import MediaBlackhole
from aiortc.mediastreams import AudioStreamTrack, MediaStreamError, MediaStreamTrack
from aiortc.mediastreams import VIDEO_CLOCK_RATE, VIDEO_TIME_BASE
from aiortc.rtcrtpsender import RTCRtpSender
from av import CodecContext, Packet
from pydub import AudioSegment
import cereal.messaging as messaging

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
  transceiver.setCodecPreferences(codec)


class EncodedBodyVideo(MediaStreamTrack):
  kind = "video"

  _start: float
  _timestamp: int

  def __init__(self):
    super().__init__()
    sock_name = 'livestreamDriverEncodeData'
    messaging.context = messaging.Context()
    self.sock = messaging.sub_sock(sock_name, None, conflate=True)
    self.pts = 0

  async def recv(self) -> Packet:
    while True:
      msg = messaging.recv_one_or_none(self.sock)
      if msg is not None:
        break
      await asyncio.sleep(0.005)

    evta = getattr(msg, msg.which())
    self.last_idx = evta.idx.encodeId

    packet = Packet(evta.header + evta.data)
    packet.time_base = VIDEO_TIME_BASE
    packet.pts = self.pts
    self.pts += 0.05 * VIDEO_CLOCK_RATE
    return packet


class WebClientSpeaker(MediaBlackhole):
  def __init__(self):
    super().__init__()
    self.p = pyaudio.PyAudio()
    self.buffer = io.BytesIO()
    self.channels = 2
    self.stream = self.p.open(format=pyaudio.paInt16, channels=self.channels, rate=48000, frames_per_buffer=9600,
                              output=True, stream_callback=self.pyaudio_callback)

  def pyaudio_callback(self, in_data, frame_count, time_info, status):
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
    for track, task in self._MediaBlackhole__tracks.items():  # pylint: disable=access-member-before-definition
      if task is None:
        self._MediaBlackhole__tracks[track] = asyncio.ensure_future(self.consume(track))

  async def stop(self):
    for task in self._MediaBlackhole__tracks.values():  # pylint: disable=access-member-before-definition
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
