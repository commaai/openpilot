import asyncio
import aiortc
import aiohttp

import argparse
import dataclasses
import json

from collections import defaultdict

from openpilot.tools.bodyteleop.webrtc.messages import Message
from openpilot.tools.bodyteleop.webrtc.connection import StreamingOffer, ConnectionProvider, StdioConnectionProvider

class VideoConsumer:
  def __init__(self):
    self.track = None

  def attach_track(self, track):
    assert track.kind == "video"
    self.track = track

  async def recv(self):
    if self.track is None:
      raise Exception("No video track")

    print("-- recv video track", self.track)
    frame = await self.track.recv()
    print("-- got frame")
    frame_arr = frame.to_ndarray(format="bgr24")
    return frame_arr
  
class AudioConsumer:
  def __init__(self):
    self.track = None

  def attach_track(self, track):
    assert track.kind == "audio"
    self.track = track

  async def recv(self):
    if self.track is None:
      raise Exception("No audio track")

    frame = await self.track.recv()
    bio = bytes(frame.planes[0])
    return bio

class DataChannelMessenger:
  def __init__(self, channel: aiortc.RTCDataChannel):
    self.channel = channel

  def send(self, message: Message):
    msg_json = json.dumps(message.to_dict())
    self.channel.send(msg_json)


class WebRTCCient:
  def __init__(self, connection_provider: ConnectionProvider):
    self.peer_connection = aiortc.RTCPeerConnection()
    self.peer_connection.on("track", self._on_track)
    self.peer_connection.on("connectionstatechange", self._on_connectionstatechange)
    self.connection_provider = connection_provider
    self.video_consumers = defaultdict(list)
    self.audio_consumers = []
    self.data_channel = None
    self.tracks_ready_event = asyncio.Event()

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
    
    self._notify_if_all_tracks_ready()

  def _on_video_track(self, track, camera_type):
    for consumer in self.video_consumers[camera_type]:
      consumer.attach_track(track)

  def _on_audio_track(self, track):
    for consumer in self.audio_consumers:
      consumer.append(track)

  def _notify_if_all_tracks_ready(self):
    all_consumers = sum(self.video_consumers.values(), []) + self.audio_consumers
    if all(consumer.track is not None for consumer in all_consumers):
      self.tracks_ready_event.set()

  def add_video_consumer(self, camera_type):
    assert camera_type in ["driver", "wideRoad", "road"]

    self.peer_connection.addTransceiver("video", direction="recvonly")
    consumer = VideoConsumer()
    self.video_consumers[camera_type].append(consumer)
    return consumer

  def add_audio_consumer(self):
    self.peer_connection.addTransceiver("audio", direction="recvonly")
    consumer = AudioConsumer()
    self.audio_consumers.append(consumer)
    return consumer

  def add_messenger(self):
    if self.data_channel:
      channel = self.data_channel
    else:
      channel = self.peer_connection.createDataChannel("data", ordered=True)
      self.data_channel = channel

    return DataChannelMessenger(channel)

  async def connect(self):
    offer = await self.peer_connection.createOffer()
    await self.peer_connection.setLocalDescription(offer)
    actual_offer = self.peer_connection.localDescription

    streaming_offer = StreamingOffer(sdp=actual_offer.sdp, 
                                           type=actual_offer.type, 
                                           video=list(self.video_consumers.keys()), 
                                           audio="audio" in self.audio_consumers)
    remote_offer = await self.connection_provider.connect(streaming_offer)
    await self.peer_connection.setRemoteDescription(remote_offer)
    # wait for the tracks to be ready
    await self.tracks_ready_event.wait()


if __name__=="__main__":
  parser = argparse.ArgumentParser(description="WebRTC client")
  parser.add_argument("cameras", metavar="CAMERA", type=str, nargs="+", default=["driver"], help="Camera types to stream")
  args = parser.parse_args()

  async def run(args):
    connection_provider = StdioConnectionProvider()
    client = WebRTCCient(connection_provider)
    consumers = {}
    for cam in args.cameras:
      video_consumer = client.add_video_consumer(cam)
      consumers[cam] = video_consumer

    await client.connect()
    while True:
      try:
        frames = await asyncio.gather(*[consumer.recv() for consumer in consumers.values()])
        for key, frame in zip(consumers.keys(), frames):
          print("Received frame from", key, frame.shape)
      except aiortc.mediastreams.MediaStreamError:
        return
      print("=====================================")

  loop = asyncio.get_event_loop()
  loop.run_until_complete(run(args))
