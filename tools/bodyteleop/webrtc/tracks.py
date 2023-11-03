#!/usr/bin/env python3

import time
import fractions
from typing import Optional

import aiortc
from aiortc.mediastreams import VIDEO_CLOCK_RATE, VIDEO_TIME_BASE
import av
import asyncio
import numpy as np
import pyaudio

from openpilot.tools.lib.framereader import FrameReader


class TiciVideoStreamTrack(aiortc.MediaStreamTrack):
  kind = "video"

  def __init__(self, camera_type: str, dt: float, time_base: fractions.Fraction = VIDEO_TIME_BASE, clock_rate: int = VIDEO_CLOCK_RATE):
    assert camera_type in ["driver", "wideRoad", "road"]
    super().__init__()
    # override track id to include camera type - client needs that for identification
    self._id = f"{camera_type}:{self._id}"
    self._dt = dt
    self._time_base = time_base
    self._clock_rate = clock_rate
    self._start = None

  async def next_pts(self, current_pts) -> float:
    pts = current_pts + self._dt * self._clock_rate

    data_time = pts * self._time_base
    if self._start is None:
      self._start = time.time() - data_time
    else:
      wait_time = self._start + data_time - time.time()
      await asyncio.sleep(wait_time)

    return pts

  def codec_preference(self) -> Optional[str]:
    return None


class DummyVideoStreamTrack(TiciVideoStreamTrack):
  def __init__(self, color: int = 0, dt: float = DT_MDL, camera_type: str = "driver"):
    super().__init__(camera_type, dt)
    self._color = color
    self._pts = 0

  async def recv(self):
    print("-- sending frame", self._pts)
    img = np.full((1920, 1080, 3), self._color, dtype=np.uint8)

    new_frame = av.VideoFrame.from_ndarray(img, format="rgb24")
    new_frame.pts = self._pts
    new_frame.time_base = self._time_base

    self._pts = await self.next_pts(self._pts)

    return new_frame


class FrameReaderVideoStreamTrack(TiciVideoStreamTrack):
  def __init__(self, input_file: str, dt: float = DT_MDL, camera_type: str = "driver"):
    super().__init__(camera_type, dt)

    frame_reader = FrameReader(input_file)
    self._frames = [frame_reader.get(i, pix_fmt="rgb24") for i in range(frame_reader.frame_count)]
    self._frame_count = len(self.frames)
    self._frame_index = 0
    self._pts = 0

  async def recv(self):
    print("-- sending frame", self._pts)
    img = self._frames[self._frame_index]

    new_frame = av.VideoFrame.from_ndarray(img, format="rgb24")
    new_frame.pts = self._pts
    new_frame.time_base = self._time_base

    self._frame_index = (self._frame_index + 1) % self._frame_count
    self._pts = await self.next_pts(self._pts)

    return new_frame

