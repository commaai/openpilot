import asyncio
import io
import numpy as np
import pyaudio
import wave

from aiortc.contrib.media import MediaBlackhole
from aiortc.mediastreams import MediaStreamError

SOUNDS = {
  'engage': '../../selfdrive/assets/sounds/engage.wav',
  'disengage': '../../selfdrive/assets/sounds/disengage.wav',
  'error': '../../selfdrive/assets/sounds/warning_immediate.wav',
}


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
