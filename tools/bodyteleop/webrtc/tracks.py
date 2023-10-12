#!/usr/bin/env python3

import argparse
import json
from typing import Awaitable, Callable, Any

import aiortc
import av
import asyncio
import numpy as np

from cereal import messaging
from openpilot.common.realtime import DT_MDL, DT_DMON
from openpilot.tools.lib.framereader import FrameReader


class TiciVideoStreamTrack(aiortc.MediaStreamTrack):
  def __init__(self, camera_type):
    super().__init__()
    # override track id to include camera type - client needs that for identification
    self._id = f"{camera_type}:{self._id}"


class LiveStreamVideoStreamTrack(TiciVideoStreamTrack):
  kind = "video"
  camera_to_sock_mapping = {
    "driver": "livestreamDriverEncodeData",
    "wideRoad": "livestreamWideRoadEncodeData",
    "road": "livestreamRoadEncodeData",
  }

  def __init__(self, camera_type):
    assert camera_type in self.camera_to_sock_mapping
    super().__init__(camera_type)

    self.sock = messaging.sub_sock(self.camera_to_sock_mapping[camera_type], conflate=True)
    self.dt = DT_MDL
    self.pts = 0

  async def recv(self):
    while True:
      msg = messaging.recv_one_or_none(self.sock)
      if msg is not None:
        break
      await asyncio.sleep(0.005)

    evta = getattr(msg, msg.which())
    self.last_idx = evta.idx.encodeId

    packet = av.Packet(evta.header + evta.data)
    packet.time_base = aiortc.mediastreams.VIDEO_TIME_BASE
    packet.pts = self.pts
    self.pts += self.dt * aiortc.mediastreams.VIDEO_CLOCK_RATE
    return packet


class FrameReaderVideoStreamTrack(TiciVideoStreamTrack):
  kind = "video"

  def __init__(self, input_file, dt=DT_MDL, camera_type="driver"):
    assert camera_type in ["driver", "wideRoad", "road"]
    super().__init__(camera_type)

    #frame_reader = FrameReader(input_file)
    #self.frames = [frame_reader.get(i, pix_fmt="rgb24") for i in range(frame_reader.frame_count)]
    shape = (1280, 720, 3) if camera_type == "driver" else (1920, 1080, 3)
    self.frames = [np.zeros(shape, dtype=np.uint8) for i in range(1200)]
    self.frame_count = len(self.frames)
    self.frame_index = 0
    self.dt = dt
    self.pts = 0

  async def recv(self):
    print("-- sending frame: ", self.frame_index)
    img = self.frames[self.frame_index]

    new_frame = av.VideoFrame.from_ndarray(img, format="rgb24")
    new_frame.pts = self.pts
    new_frame.time_base = aiortc.mediastreams.VIDEO_TIME_BASE

    self.frame_index = (self.frame_index + 1) % self.frame_count
    self.pts += self.dt * aiortc.mediastreams.VIDEO_CLOCK_RATE

    return new_frame


import pyaudio
from pydub import AudioSegment


class AudioInputStreamTrack(aiortc.mediastreams.AudioStreamTrack):
  def __init__(self, format=pyaudio.paInt16, rate=16000, channels=1, packet_time=0.020, device_index=None):
    super().__init__()

    self.p = pyaudio.PyAudio()
    frame_per_buffer = packet_time * rate
    self.stream = self.p.open(format=format, 
                              channels=channels, 
                              rate=rate, 
                              frames_per_buffer=frame_per_buffer, 
                              input=True, 
                              input_device_index=device_index)
    self.format = format
    self.rate = rate
    self.channels = channels
    self.packet_time = packet_time
    self.chunk_size = int(rate * packet_time)
    self.chunk_index = 0
    self.pts = 0

  async def recv(self):
    mic_data = self.stream.read(self.chunk_size)
    mic_sound = AudioSegment(mic_data, sample_width=pyaudio.get_sample_size(self.format), channels=self.channels, frame_rate=self.rate)
    # create stereo sound?
    mic_sound = AudioSegment.from_mono_audiosegments(mic_sound, mic_sound)
    packet = av.Packet(mic_sound.raw_data)
    # TODO
    return None
