import asyncio
import io
import socket

import aiortc
import av
import numpy as np
import pyaudio


class AudioInputStreamTrack(aiortc.mediastreams.AudioStreamTrack):
  PYAUDIO_TO_AV_FORMAT_MAP = {
      pyaudio.paUInt8: 'u8',
      pyaudio.paInt16: 's16',
      pyaudio.paInt24: 's24',
      pyaudio.paInt32: 's32',
      pyaudio.paFloat32: 'flt',
  }

  def __init__(self, audio_format: int = pyaudio.paInt16, rate: int = 16000, channels: int = 1, packet_time: float = 0.020, device_index: int | None = None):
    super().__init__()

    self.p = pyaudio.PyAudio()
    chunk_size = int(packet_time * rate)
    self.stream = self.p.open(format=audio_format,
                              channels=channels,
                              rate=rate,
                              frames_per_buffer=chunk_size,
                              input=True,
                              input_device_index=device_index)
    self.format = audio_format
    self.rate = rate
    self.channels = channels
    self.packet_time = packet_time
    self.chunk_size = chunk_size
    self.pts = 0

  async def recv(self):
    mic_data = self.stream.read(self.chunk_size)
    mic_array = np.frombuffer(mic_data, dtype=np.int16)
    mic_array = np.expand_dims(mic_array, axis=0)
    layout = 'stereo' if self.channels > 1 else 'mono'
    frame = av.AudioFrame.from_ndarray(mic_array, format=self.PYAUDIO_TO_AV_FORMAT_MAP[self.format], layout=layout)
    frame.rate = self.rate
    frame.pts = self.pts
    self.pts += frame.samples

    return frame


class AudioOutputSpeaker:
  def __init__(self, audio_format: int = pyaudio.paInt16, rate: int = 48000, channels: int = 2, packet_time: float = 0.2, device_index: int | None = None):

    chunk_size = int(packet_time * rate)
    self.p = pyaudio.PyAudio()
    self.buffer = io.BytesIO()
    self.channels = channels
    self.stream = self.p.open(format=audio_format,
                              channels=channels,
                              rate=rate,
                              frames_per_buffer=chunk_size,
                              output=True,
                              output_device_index=device_index,
                              stream_callback=self.__pyaudio_callback)
    self.tracks_and_tasks: list[tuple[aiortc.MediaStreamTrack, asyncio.Task | None]] = []

  def __pyaudio_callback(self, in_data, frame_count, time_info, status):
    if self.buffer.getbuffer().nbytes < frame_count * self.channels * 2:
      buff = b'\x00\x00' * frame_count * self.channels
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

  async def __consume(self, track):
    while True:
      try:
        frame = await track.recv()
      except aiortc.MediaStreamError:
        return

      self.buffer.write(bytes(frame.planes[0]))

  def hasTrack(self, track: aiortc.MediaStreamTrack) -> bool:
    return any(t == track for t, _ in self.tracks_and_tasks)

  def addTrack(self, track: aiortc.MediaStreamTrack):
    if not self.hasTrack(track):
      self.tracks_and_tasks.append((track, None))

  def start(self):
    for index, (track, task) in enumerate(self.tracks_and_tasks):
      if task is None:
        self.tracks_and_tasks[index] = (track, asyncio.create_task(self.__consume(track)))

  def stop(self):
    for _, task in self.tracks_and_tasks:
      if task is not None:
        task.cancel()

    self.tracks_and_tasks = []
    self.stream.stop_stream()
    self.stream.close()
    self.p.terminate()


class CerealAudioStreamTrack(aiortc.mediastreams.AudioStreamTrack):
  """Reads mic audio from micd's rawAudioData cereal topic."""
  def __init__(self, rate: int = 16000):
    super().__init__()
    from cereal import messaging
    self.sm = messaging.SubMaster(['rawAudioData'])
    self.rate = rate
    self.buffer = bytearray()
    self.pts = 0
    self.bytes_per_frame = int(0.020 * rate) * 2

  async def recv(self):
    while len(self.buffer) < self.bytes_per_frame:
      self.sm.update(0)
      if self.sm.updated['rawAudioData']:
        self.buffer.extend(self.sm['rawAudioData'].data)
      else:
        await asyncio.sleep(0.005)

    chunk = self.buffer[:self.bytes_per_frame]
    self.buffer = self.buffer[self.bytes_per_frame:]

    frame = av.AudioFrame.from_ndarray(
      np.frombuffer(chunk, dtype=np.int16).reshape(1, -1),
      format='s16', layout='mono'
    )
    frame.rate = self.rate
    frame.pts = self.pts
    self.pts += frame.samples
    return frame


class SocketAudioOutput:
  """Sends incoming WebRTC audio to soundd via UDP."""
  def __init__(self, rate: int = 48000):
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self.dest = ('127.0.0.1', 27000)
    self.resampler = av.AudioResampler(format='s16', layout='mono', rate=rate)
    self.task = None
    self.track = None

  async def _consume(self):
    while True:
      try:
        frame = await self.track.recv()
        for r_frame in self.resampler.resample(frame):
          self.sock.sendto(bytes(r_frame.planes[0]), self.dest)
      except aiortc.MediaStreamError:
        return

  def addTrack(self, track: aiortc.MediaStreamTrack):
    self.track = track

  def hasTrack(self, track: aiortc.MediaStreamTrack) -> bool:
    return self.track == track

  def start(self):
    if self.track and not self.task:
      self.task = asyncio.create_task(self._consume())

  def stop(self):
    if self.task:
      self.task.cancel()
    self.sock.close()
