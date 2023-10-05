import aiortc
import aiohttp

class WebRTCCient:
  def __init__(self, address="127.0.0.1", port=8080, endpoint="livestream_driver"):
    self.peer_connection = aiortc.RTCPeerConnection()
    self.peer_connection.on("track", self.on_track)

  def on_track(self, track):
    if track.kind == "video":
      self.peer_connection.addTrack(track)
    elif track.kind == "audio":
      self.peer_connection.addTrack(track)

  async def connect(self):
    offer = self.peer_connection.createOffer()
    await self.peer_connection.setLocalDescription(offer)
    with aiohttp.ClientSession() as session:
      answer = await session.get(f"http://{self.address}:{self.port}/{self.endpoint}", json=offer)
      # TODO

  def recv(self, timeout_ms=100):
    # TODO
    pass

  def is_connected(self):
    # TODO
    return False

