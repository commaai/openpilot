import asyncio
from typing import Optional

import aiortc
import av
import numpy as np
import pyaudio

from cereal import messaging
from openpilot.tools.bodyteleop.webrtc.tracks import TiciVideoStreamTrack
from openpilot.tools.lib.framereader import FrameReader
from openpilot.common.realtime import DT_MDL, DT_DMON


class LiveStreamVideoStreamTrack(TiciVideoStreamTrack):
  camera_to_sock_mapping = {
    "driver": "livestreamDriverEncodeData",
    "wideRoad": "livestreamWideRoadEncodeData",
    "road": "livestreamRoadEncodeData",
  }

  def __init__(self, camera_type: str):
    dt = DT_DMON if camera_type == "driver" else DT_MDL
    super().__init__(camera_type, dt)

    self._sock = messaging.sub_sock(self.camera_to_sock_mapping[camera_type], conflate=True)
    self._pts = 0

  async def recv(self):
    while True:
      msg = messaging.recv_one_or_none(self._sock)
      if msg is not None:
        break
      await asyncio.sleep(0.005)

    evta = getattr(msg, msg.which())

    packet = av.Packet(evta.header + evta.data)
    packet.time_base = self._time_base
    packet.pts = self._pts

    self.log_debug("track sending frame %s", self._pts)
    self._pts += self._dt * self._clock_rate

    return packet

  def codec_preference(self) -> Optional[str]:
    return "H264"


class AudioInputStreamTrack(aiortc.mediastreams.AudioStreamTrack):
  PYAUDIO_TO_AV_FORMAT_MAP = {
      pyaudio.paUInt8: 'u8',
      pyaudio.paInt16: 's16',
      pyaudio.paInt24: 's24',
      pyaudio.paInt32: 's32',
      pyaudio.paFloat32: 'flt',
  }

  def __init__(self, audio_format: int = pyaudio.paInt16, rate: int = 16000, channels: int = 1, packet_time: float = 0.020, device_index: Optional[int] = None):
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


class FrameReaderVideoStreamTrack(TiciVideoStreamTrack):
  def __init__(self, input_file: str, dt: float = DT_MDL, camera_type: str = "driver"):
    super().__init__(camera_type, dt)

    frame_reader = FrameReader(input_file)
    self._frames = [frame_reader.get(i, pix_fmt="rgb24") for i in range(frame_reader.frame_count)]
    self._frame_count = len(self.frames)
    self._frame_index = 0
    self._pts = 0

  async def recv(self):
    self.log_debug("track sending frame %s", self._pts)
    img = self._frames[self._frame_index]

    new_frame = av.VideoFrame.from_ndarray(img, format="rgb24")
    new_frame.pts = self._pts
    new_frame.time_base = self._time_base

    self._frame_index = (self._frame_index + 1) % self._frame_count
    self._pts = await self.next_pts(self._pts)

    return new_frame
