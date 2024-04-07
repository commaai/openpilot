import asyncio
import logging
import time
import fractions
from typing import Any

import aiortc
from aiortc.mediastreams import VIDEO_CLOCK_RATE, VIDEO_TIME_BASE


def video_track_id(camera_type: str, track_id: str) -> str:
  return f"{camera_type}:{track_id}"


def parse_video_track_id(track_id: str) -> tuple[str, str]:
  parts = track_id.split(":")
  if len(parts) != 2:
    raise ValueError(f"Invalid video track id: {track_id}")

  camera_type, track_id = parts
  return camera_type, track_id


class TiciVideoStreamTrack(aiortc.MediaStreamTrack):
  """
  Abstract video track which associates video track with camera_type
  """
  kind = "video"

  def __init__(self, camera_type: str, dt: float, time_base: fractions.Fraction = VIDEO_TIME_BASE, clock_rate: int = VIDEO_CLOCK_RATE):
    assert camera_type in ["driver", "wideRoad", "road"]
    super().__init__()
    # override track id to include camera type - client needs that for identification
    self._id: str = video_track_id(camera_type, self._id)
    self._dt: float = dt
    self._time_base: fractions.Fraction = time_base
    self._clock_rate: int = clock_rate
    self._start: float | None = None
    self._logger = logging.getLogger("WebRTCStream")

  def log_debug(self, msg: Any, *args):
    self._logger.debug(f"{type(self)}() {msg}", *args)

  async def next_pts(self, current_pts) -> float:
    pts: float = current_pts + self._dt * self._clock_rate

    data_time = pts * self._time_base
    if self._start is None:
      self._start = time.time() - data_time
    else:
      wait_time = self._start + data_time - time.time()
      await asyncio.sleep(wait_time)

    return pts

  def codec_preference(self) -> str | None:
    return None


class TiciTrackWrapper(aiortc.MediaStreamTrack):
  """
  Associates video track with camera_type
  """
  def __init__(self, camera_type: str, track: aiortc.MediaStreamTrack):
    assert track.kind == "video"
    assert not isinstance(track, TiciVideoStreamTrack)
    super().__init__()
    self._id = video_track_id(camera_type, track.id)
    self._track = track

  @property
  def kind(self) -> str:
    return self._track.kind

  async def recv(self):
    return await self._track.recv()
