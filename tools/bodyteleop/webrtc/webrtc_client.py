import aiortc
import aiohttp

class WebRTCCient:
  def __init__(self, address="127.0.0.1", port=8080, endpoint="livestream_driver"):
    self.peer_connection = aiortc.RTCPeerConnection()
    self.peer_connection.on("track", self.on_track)
    self.video_track = None

  def on_track(self, track):
    if track.kind == "video":
      self.video_track = track
    elif track.kind == "audio":
      pass

  async def connect(self):
    offer = self.peer_connection.createOffer()
    await self.peer_connection.setLocalDescription(offer)
    async with aiohttp.ClientSession() as session:
      response = await session.get(f"http://{self.address}:{self.port}/{self.endpoint}", json=offer)
      async with response:
        if response.status != 200:
          raise Exception(f"Offer request failed with HTTP status code {response.status}")
        answer = await response.json()
        remote_offer = aiortc.RTCSessionDescription(sdp=answer.sdp, type=answer.type)
        self.peer_connection.setRemoteDescription(remote_offer)

  async def recv(self, timeout_ms=100):
    if self.video_track is None:
      raise Exception("No video track")

    frame = await self.video_track.recv()
    frame_arr = frame.to_ndarray(format="bgr24")
    return frame_arr

  def is_connected(self):
    return self.video_track is not None

