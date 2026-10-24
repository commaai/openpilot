import fractions
import logging
import uuid
from typing import Any, Tuple


VIDEO_CLOCK_RATE = 90000
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)


def video_track_id(camera_type: str, track_id: str) -> str:
  return f"{camera_type}:{track_id}"


def parse_video_track_id(track_id: str) -> Tuple[str, str]:
  parts = track_id.split(":")
  if len(parts) != 2:
    raise ValueError(f"Invalid video track id: {track_id}")

  camera_type, track_id = parts
  return camera_type, track_id


class TiciVideoStreamTrack:
  """
  Abstract video track which associates video track with camera_type.
  """
  kind = "video"

  def __init__(self, camera_type: str, dt: float, time_base: fractions.Fraction = VIDEO_TIME_BASE, clock_rate: int = VIDEO_CLOCK_RATE):
    assert camera_type in ["driver", "wideRoad", "road"]
    self._id: str = video_track_id(camera_type, str(uuid.uuid4()))
    self._time_base: fractions.Fraction = time_base
    self._clock_rate: int = clock_rate
    self._logger = logging.getLogger("WebRTCStream")
    self.readyState = "live"

  @property
  def id(self) -> str:
    return self._id

  def stop(self) -> None:
    self.readyState = "ended"

  def log_debug(self, msg: Any, *args):
    self._logger.debug(f"{type(self)}() {msg}", *args)

  async def recv(self):
    raise NotImplementedError()

  def request_keyframe(self) -> None:
    pass


class TiciTrackWrapper(TiciVideoStreamTrack):
  """
  Associates a generic video track with camera_type.
  """
  def __init__(self, camera_type: str, track: Any):
    assert track.kind == "video"
    super().__init__(camera_type, getattr(track, "_dt", 0.05))
    self._id = video_track_id(camera_type, track.id)
    self._track = track

  async def recv(self):
    return await self._track.recv()

  def stop(self) -> None:
    super().stop()
    if hasattr(self._track, "stop"):
      self._track.stop()
