import asyncio
import aiortc
import aiohttp

import argparse
import dataclasses
import json

from collections import defaultdict

@dataclasses.dataclass
class WebRTCStreamingOffer:
  sdp: str
  type: str
  video: list[str]
  audio: bool

class WebRTCVideoClient:
  def __init__(self):
    self.video_track = None

  def attach_track(self, track):
    assert track.kind == "video"
    self.video_track = track

  async def recv(self):
    if self.video_track is None:
      raise Exception("No video track")

    print("-- recv video track", self.video_track)
    frame = await self.video_track.recv()
    print("-- got frame")
    frame_arr = frame.to_ndarray(format="bgr24")
    return frame_arr
  
class WebRTCAudioClient:
  def __init__(self):
    self.audio_track = None

  def attach_track(self, track):
    assert track.kind == "audio"
    self.audio_track = track

  async def recv(self):
    if self.audio_track is None:
      raise Exception("No audio track")

    frame = await self.audio_track.recv()
    bio = bytes(frame.planes[0])
    return bio

class WebRTCConnectionProvider:
  async def connect(self, offer) -> aiortc.RTCSessionDescription:
    raise NotImplementedError()
  
class WebRTCStdInConnectionProvider(WebRTCConnectionProvider):
  async def connect(self, offer: WebRTCStreamingOffer) -> aiortc.RTCSessionDescription:
    async def async_input():
      return await asyncio.to_thread(input)

    print("-- Please send this JSON to server --")
    print(json.dumps(dataclasses.asdict(offer)))
    print("-- Press enter when the answer is ready --")
    raw_payload = await async_input()
    payload = json.loads(raw_payload)
    answer = aiortc.RTCSessionDescription(**payload)

    return answer
  
class WebRTCHTTPConnectionProvider(WebRTCConnectionProvider):
  def __init__(self, address="127.0.0.1", port=8080):
    self.address = address
    self.port = port

  async def connect(self, offer: WebRTCStreamingOffer) -> aiortc.RTCSessionDescription:
    payload = dataclasses.asdict(offer)
    async with aiohttp.ClientSession() as session:
      response = await session.get(f"http://{self.address}:{self.port}/webrtc", json=payload)
      async with response:
        if response.status != 200:
          raise Exception(f"Offer request failed with HTTP status code {response.status}")
        answer = await response.json()
        remote_offer = aiortc.RTCSessionDescription(sdp=answer.sdp, type=answer.type)

        return remote_offer

class WebRTCCient:
  def __init__(self, connection_provider: WebRTCConnectionProvider):
    self.peer_connection = aiortc.RTCPeerConnection()
    self.peer_connection.on("track", self._on_track)
    self.peer_connection.on("connectionstatechange", self._on_connectionstatechange)
    self.connection_provider = connection_provider
    self.video_consumers = defaultdict(list)
    self.audio_consumers = []
    self.channels = {}

  def _on_connectionstatechange(self):
    print("-- connection state is", self.peer_connection.connectionState)

  def _on_track(self, track):
    print("-- got track: ", track.kind, track.id)
    if track.kind == "video":
      # format: "camera_type:camera_id"
      parts = track.id.split(":")
      if len(parts) < 2:
        return

      camera_type = parts[0]
      self._on_video_track(track, camera_type)
    elif track.kind == "audio":
      self._on_audio_track(track)

  def _on_video_track(self, track, camera_type):
    for consumer in self.video_consumers[camera_type]:
      consumer.attach_track(track)

  def _on_audio_track(self, track):
    for consumer in self.audio_consumers:
      consumer.append(track)

  def add_video_consumer(self, camera_type):
    assert camera_type in ["driver", "wideRoad", "road"]

    self.peer_connection.addTransceiver("video", direction="recvonly")
    consumer = WebRTCVideoClient()
    self.video_consumers[camera_type].append(consumer)
    return consumer

  def add_audio_consumer(self):
    self.peer_connection.addTransceiver("audio", direction="recvonly")
    consumer = WebRTCAudioClient()
    self.audio_consumers.append(consumer)
    return consumer

  def add_channel_producer(self, channel_name):
    if channel_name in self.channels:
      raise Exception(f"Channel {channel_name} already exists")

    channel = self.peer_connection.createDataChannel(channel_name)
    self.channels[channel_name] = channel
    return channel

  async def connect(self):
    offer = await self.peer_connection.createOffer()
    await self.peer_connection.setLocalDescription(offer)
    actual_offer = self.peer_connection.localDescription

    streaming_offer = WebRTCStreamingOffer(sdp=actual_offer.sdp, 
                                           type=actual_offer.type, 
                                           video=list(self.video_consumers.keys()), 
                                           audio="audio" in self.audio_consumers)
    remote_offer = await self.connection_provider.connect(streaming_offer)
    await self.peer_connection.setRemoteDescription(remote_offer)


if __name__=="__main__":
  parser = argparse.ArgumentParser(description="WebRTC client")
  parser.add_argument("cameras", metavar="CAMERA", type=str, nargs="+", default=["driver"], help="Camera types to stream")
  args = parser.parse_args()

  async def run(args):
    connection_provider = WebRTCStdInConnectionProvider()
    client = WebRTCCient(connection_provider)
    consumers = {}
    for cam in args.cameras:
      video_consumer = client.add_video_consumer(cam)
      consumers[cam] = video_consumer

    await client.connect()
    while True:
      try:
        await asyncio.sleep(5)
        frames = await asyncio.gather(*[consumer.recv() for consumer in consumers.values()])
        for key, frame in zip(consumers.keys(), frames):
          print("Received frame from", key, frame.shape)
      except aiortc.mediastreams.MediaStreamError:
        return
      print("=====================================")

  loop = asyncio.get_event_loop()
  loop.run_until_complete(run(args))
