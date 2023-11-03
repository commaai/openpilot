import time
import fractions
from typing import Optional

import aiortc
from aiortc.mediastreams import VIDEO_CLOCK_RATE, VIDEO_TIME_BASE
import av
import asyncio
import numpy as np


class TiciVideoStreamTrack(aiortc.MediaStreamTrack):
  kind = "video"

  def __init__(self, camera_type: str, dt: float, time_base: fractions.Fraction = VIDEO_TIME_BASE, clock_rate: int = VIDEO_CLOCK_RATE):
    assert camera_type in ["driver", "wideRoad", "road"]
    super().__init__()
    # override track id to include camera type - client needs that for identification
    self._id: str = f"{camera_type}:{self._id}"
    self._dt: float = dt
    self._time_base: fractions.Fraction = time_base
    self._clock_rate: int = clock_rate
    self._start: Optional[float] = None

  async def next_pts(self, current_pts) -> float:
    pts: float = current_pts + self._dt * self._clock_rate

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
  def __init__(self, dt: float, camera_type: str, color: int = 0):
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
