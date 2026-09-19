import asyncio
from dataclasses import dataclass
import struct
import time

from teleoprtc.tracks import TiciVideoStreamTrack

from openpilot.cereal import messaging
from openpilot.common.realtime import DT_MDL
from openpilot.common.camera120 import CAMERA_FPS, camera120_enabled
from openpilot.common.params import Params


# v4l2 buffer flag marking an encoded keyframe (linux/videodev2.h)
V4L2_BUF_FLAG_KEYFRAME = 0x8

# arbitrary 16-byte UUID identifying openpilot frame-timing SEI messages
TIMING_SEI_UUID = bytes([
  0xa5, 0xe0, 0xc4, 0xa4, 0x5b, 0x6e, 0x4e, 0x1e,
  0x9c, 0x7e, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc,
])
_SEI_PREFIX = b'\x00\x00\x00\x01\x06\x05\x30' + TIMING_SEI_UUID


@dataclass(frozen=True)
class EncodedVideoFrame:
  data: bytes
  pts: int

  def __bytes__(self) -> bytes:
    return self.data


class LiveStreamVideoStreamTrack(TiciVideoStreamTrack):
  camera_to_sock_mapping = {
    "driver": "livestreamCabinEncodeData",
    "wideRoad": "livestreamWideRoadEncodeData",
    "road": "livestreamNarrowRoadEncodeData",
  }

  def __init__(self, camera_type: str, video_enabled: bool = True):
    self.high_fps = camera120_enabled()
    # CVO specifies clockwise quarter-turns: 3 means 90 degrees counterclockwise.
    # Keep the sensor and H.264 pixels unchanged; the receiver applies rotation.
    self.video_orientation = 3 if self.high_fps else 0
    self.h264_profile_level_id = "42e020" if self.high_fps else None  # Constrained Baseline profile, Level 3.2
    self.h264_allow_lower_level = self.high_fps
    super().__init__(camera_type, 1 / CAMERA_FPS if self.high_fps else DT_MDL)

    self._sock = self._make_sock(camera_type)
    self._pts = 0
    self._t0_ns = time.monotonic_ns()
    self.timing_sei_enabled = False
    self.params = Params()
    self._seen_keyframe = False
    self._last_frame_id = None
    self._capture_t0_ns = None
    self.video_enabled = video_enabled

  def stop(self) -> None:
    super().stop()
    self._sock = None

  def _make_sock(self, camera_type: str) -> messaging.SubSocket:
    # The bench has only one physical stream, including when a client selects wide/driver.
    source = "road" if self.high_fps else camera_type
    return messaging.sub_sock(self.camera_to_sock_mapping[source], conflate=not self.high_fps)

  def switch_camera(self, camera_type: str) -> None:
    self._sock = self._make_sock(camera_type)
    self._seen_keyframe = False
    self._last_frame_id = None
    self.request_keyframe()

  def enable(self, enabled: bool):
    self.video_enabled = enabled
    if not enabled:
      self._seen_keyframe = False

  def request_keyframe(self) -> None:
    self.params.put("LivestreamRequestKeyframe", True, block=False)

  def _build_frame_data(self, msg) -> bytes:
    encode_data = getattr(msg, msg.which())
    if not self.timing_sei_enabled:
      return encode_data.header + encode_data.data

    idx = encode_data.idx
    sei_nal = _SEI_PREFIX + struct.pack('>4d',
      (idx.timestampEof - idx.timestampSof) / 1e6,
      (msg.logMonoTime - idx.timestampEof) / 1e6,
      (time.monotonic_ns() - msg.logMonoTime) / 1e6,
      time.time() * 1000,  # noqa: TID251
    ) + b'\x80'
    return encode_data.header + sei_nal + encode_data.data

  async def recv(self):
    while True:
      # while video is disabled, pause here without returning
      if not self.video_enabled:
        await asyncio.sleep(0.005)
        continue

      msg = messaging.recv_one_or_none(self._sock)
      if msg is not None:
        idx = getattr(msg, msg.which()).idx
        if self.high_fps:
          if self._last_frame_id is not None and idx.frameId != self._last_frame_id + 1:
            self._seen_keyframe = False
            self.request_keyframe()
          self._last_frame_id = idx.frameId
          if not self._seen_keyframe and not (idx.flags & V4L2_BUF_FLAG_KEYFRAME):
            await asyncio.sleep(0)
            continue
        if not self._seen_keyframe and (getattr(msg, msg.which()).idx.flags & V4L2_BUF_FLAG_KEYFRAME):
          self._seen_keyframe = True
          self.params.put("LivestreamRequestKeyframe", False, block=False)
        break
      await asyncio.sleep(0.001 if self.high_fps else 0.005)

    if self.high_fps:
      capture_ns = getattr(msg, msg.which()).idx.timestampSof
      if self._capture_t0_ns is None:
        self._capture_t0_ns = capture_ns
      self._pts = max(self._pts + 1, (capture_ns - self._capture_t0_ns) * self._clock_rate // 1_000_000_000)
    else:
      self._pts = ((time.monotonic_ns() - self._t0_ns) * self._clock_rate) // 1_000_000_000
    self.log_debug("track sending frame %d", self._pts)

    return EncodedVideoFrame(self._build_frame_data(msg), self._pts)
