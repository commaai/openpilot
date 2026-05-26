import asyncio
import struct
import time

import av
from teleoprtc.tracks import TiciVideoStreamTrack

from cereal import messaging
from openpilot.common.realtime import DT_MDL, DT_DMON

# arbitrary 16-byte UUID identifying openpilot frame-timing SEI messages
TIMING_SEI_UUID = bytes([
  0xa5, 0xe0, 0xc4, 0xa4, 0x5b, 0x6e, 0x4e, 0x1e,
  0x9c, 0x7e, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc,
])
_SEI_PREFIX = b'\x00\x00\x00\x01\x06\x05\x38' + TIMING_SEI_UUID


class LiveStreamVideoStreamTrack(TiciVideoStreamTrack):
  camera_to_sock_mapping = {
    "driver": "livestreamDriverEncodeData",
    "wideRoad": "livestreamWideRoadEncodeData",
    "road": "livestreamRoadEncodeData",
  }
  camera_to_state_sock_mapping = {
    "driver": "driverCameraState",
    "wideRoad": "wideRoadCameraState",
    "road": "roadCameraState",
  }

  def __init__(self, camera_type: str):
    dt = DT_DMON if camera_type == "driver" else DT_MDL
    super().__init__(camera_type, dt)

    self._sock = self._make_sock(camera_type)
    self._state_sock = self._make_state_sock(camera_type)
    self._processing_times: dict[int, float] = {}
    self._pts = 0
    self._t0_ns = time.monotonic_ns()
    self.timing_sei_enabled = False

  def _make_sock(self, camera_type: str) -> messaging.SubSocket:
    return messaging.sub_sock(self.camera_to_sock_mapping[camera_type], conflate=True)

  def _make_state_sock(self, camera_type: str) -> messaging.SubSocket:
    return messaging.sub_sock(self.camera_to_state_sock_mapping[camera_type], conflate=False)

  def switch_camera(self, camera_type: str) -> None:
    self._sock = self._make_sock(camera_type)
    self._state_sock = self._make_state_sock(camera_type)
    self._processing_times.clear()

  def _build_frame_data(self, msg) -> bytes:
    encode_data = getattr(msg, msg.which())
    idx = encode_data.idx

    for state_msg in messaging.drain_sock(self._state_sock):
      state = getattr(state_msg, state_msg.which())
      self._processing_times[state.frameId] = state.processingTime
    isp_ms = self._processing_times.pop(idx.frameId, 0.0) * 1e3
    self._processing_times = {fid: pt for fid, pt in self._processing_times.items() if fid > idx.frameId}

    sensor_ms = (idx.timestampEof - idx.timestampSof) / 1e6
    encode_ms = max((msg.logMonoTime - idx.timestampEof) / 1e6 - isp_ms, 0.0)
    transit_ms = (time.monotonic_ns() - msg.logMonoTime) / 1e6
    if idx.encodeId % 20 == 0:
      print(f"encode times frame={idx.encodeId} sensor={sensor_ms:.2f}ms "
            f"isp={isp_ms:.2f}ms encode={encode_ms:.2f}ms transit={transit_ms:.2f}ms")

    if not self.timing_sei_enabled:
      return encode_data.header + encode_data.data

    sei_nal = _SEI_PREFIX + struct.pack('>5d',
      sensor_ms,
      isp_ms,
      encode_ms,
      transit_ms,
      time.time() * 1000,  # noqa: TID251
    ) + b'\x80'
    return encode_data.header + sei_nal + encode_data.data

  async def recv(self):
    while True:
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
