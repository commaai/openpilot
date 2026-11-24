import asyncio
from contextlib import aclosing
import io
import logging
import struct
import time

from aiohttp import ClientTimeout
from aiortc import RTCConfiguration, RTCPeerConnection, RTCRtpSender, RTCSessionDescription

# User-data UUID in openpilot.system.webrtc.device.video.
TIMING_SEI_UUID = bytes.fromhex("a5e0c4a45b6e4e1e9c7e123456789abc")


def capture_time(frame):
  for side in frame.side_data[:]:
    if side.type.name == "SEI_UNREGISTERED":
      data = bytes(side)
      if len(data) == 48 and data.startswith(TIMING_SEI_UUID):
        sensor, encoder, sender, unix_ms = struct.unpack_from(">4d", data, 16)
        return (unix_ms - sensor - encoder - sender) / 1000
  return None


def encode_jpeg(frame):
  # Restore the game's calibrated wide-camera raster after the hardware encoder's 720p resize.
  out = io.BytesIO()
  frame.to_image(width=1008, height=570).save(out, "JPEG", quality=75)
  return out.getvalue()


async def _frames(http, url, headers):
  peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
  try:
    video = peer.addTransceiver("video", direction="recvonly")
    video.setCodecPreferences([codec for codec in RTCRtpSender.getCapabilities("video").codecs if codec.mimeType == "video/H264"])
    async with asyncio.timeout(30):
      await peer.setLocalDescription(await peer.createOffer())
      async with http.post(url + "/camera/offer", json={"sdp": peer.localDescription.sdp}, headers=headers, timeout=ClientTimeout(total=25)) as response:
        response.raise_for_status()
        answer = await response.json()
      await peer.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type="answer"))
    while True:
      async with asyncio.timeout(10):
        while True:
          frame = await video.receiver.track.recv()
          captured = capture_time(frame)
          if captured is not None and 0 <= time.time() - captured < .2:  # noqa: TID251
            break
      yield captured, await asyncio.to_thread(encode_jpeg, frame)
  finally:
    await peer.close()


async def camera_frames(http, url, headers=None):
  previous = 0.
  while True:
    try:
      async with aclosing(_frames(http, url.rstrip("/"), headers)) as frames:
        async for captured, jpeg in frames:
          if captured > previous and 0 <= time.time() - captured < .2:  # noqa: TID251
            previous = captured
            yield captured, jpeg
    except Exception as e:
      logging.getLogger("rtc-camera").warning("Wide-camera stream disconnected: %r", e)
      await asyncio.sleep(1)
