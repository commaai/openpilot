import asyncio

import av
from teleoprtc.tracks import TiciVideoStreamTrack

from cereal import messaging
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

  def codec_preference(self) -> str | None:
    return "H264"
