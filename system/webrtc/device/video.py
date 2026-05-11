import asyncio
import time

import av
from teleoprtc.tracks import TiciVideoStreamTrack

from cereal import messaging
from openpilot.common.realtime import DT_MDL, DT_DMON


class LiveStreamVideoStreamTrack(TiciVideoStreamTrack):
  def __init__(self, camera_type: str):
    dt = DT_DMON if camera_type == "driver" else DT_MDL
    super().__init__(camera_type, dt)

    self._sock = messaging.sub_sock("livestreamCameraEncodeData", conflate=True)
    self._t0_ns = time.monotonic_ns()

  async def recv(self):
    while True:
      msg = messaging.recv_one_or_none(self._sock)
      if msg is not None:
        break
      await asyncio.sleep(0.005)

    encode_data = getattr(msg, msg.which())

    packet = av.Packet(encode_data.header + encode_data.data)
    packet.time_base = self._time_base

    pts = ((time.monotonic_ns() - self._t0_ns) * self._clock_rate) // 1_000_000_000
    packet.pts = pts
    self.log_debug("track sending frame %d", pts)
    return packet

  def codec_preference(self) -> str | None:
    return "H264"
