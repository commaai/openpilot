import asyncio
import struct
import time

import av
from teleoprtc.tracks import TiciVideoStreamTrack
from aiortc import MediaStreamError

from cereal import messaging
from openpilot.common.realtime import DT_MDL, DT_DMON
from openpilot.common.params import Params


# v4l2 buffer flag marking an encoded keyframe (linux/videodev2.h)
V4L2_BUF_FLAG_KEYFRAME = 0x8

# arbitrary 16-byte UUID identifying openpilot frame-timing SEI messages
TIMING_SEI_UUID = bytes([
  0xa5, 0xe0, 0xc4, 0xa4, 0x5b, 0x6e, 0x4e, 0x1e,
  0x9c, 0x7e, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc,
])
_SEI_PREFIX = b'\x00\x00\x00\x01\x06\x05\x30' + TIMING_SEI_UUID


class LiveStreamVideoStreamTrack(TiciVideoStreamTrack):
  camera_to_sock_mapping = {
    "driver": "livestreamDriverEncodeData",
    "wideRoad": "livestreamWideRoadEncodeData",
    "road": "livestreamRoadEncodeData",
  }

  def __init__(self, camera_type: str, video_enabled: bool):
    dt = DT_DMON if camera_type == "driver" else DT_MDL
    super().__init__(camera_type, dt)

    self._sock = self._make_sock(camera_type)
    self._pts = 0
    self._t0_ns = time.monotonic_ns()
    self.timing_sei_enabled = False
    self.params = Params()
    self.seen_keyframe = False
    self.video_enabled = video_enabled

  def stop(self) -> None:
    super().stop()
    self._sock = None

  def _make_sock(self, camera_type: str) -> messaging.SubSocket:
    return messaging.sub_sock(self.camera_to_sock_mapping[camera_type], conflate=True)

  def switch_camera(self, camera_type: str) -> None:
    self._sock = self._make_sock(camera_type)

  def enable_video(self, enabled: bool):
    self.video_enabled = enabled
    if not enabled: self.seen_keyframe = False

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
      if self.readyState != "live":
        raise MediaStreamError

      # while video is disabled, pause here without returning
      if not self.video_enabled:
        await asyncio.sleep(0.005)
        continue

      msg = messaging.recv_one_or_none(self._sock)
      if msg is not None:
        break
      await asyncio.sleep(0.005)

    packet = av.Packet(self._build_frame_data(msg))
    packet.time_base = self._time_base

    self._pts =  ((time.monotonic_ns() - self._t0_ns) * self._clock_rate) // 1_000_000_000
    packet.pts = self._pts
    self.log_debug("track sending frame %d", self._pts)

    return packet

  def codec_preference(self) -> str | None:
    return "H264"
