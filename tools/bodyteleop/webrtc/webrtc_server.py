#!/usr/bin/env python3

import argparse
import json

import aiortc
from aiortc.mediastreams import VIDEO_CLOCK_RATE, VIDEO_TIME_BASE
import av
import asyncio
import numpy as np

from cereal import messaging
from openpilot.common.realtime import DT_MDL, DT_DMON
from openpilot.tools.bodyteleop.webrtc.webrtc_client import WebRTCStreamingMetadata, WebRTCStreamingOffer
from openpilot.tools.lib.framereader import FrameReader


class TiciVideoStreamTrack(aiortc.MediaStreamTrack):
  def __init__(self, camera_type):
    super().__init__()
    # override track id to include camera type - client needs that for identification
    self._id = f"{camera_type}:{self._id}"


class LiveStreamVideoStreamTrack(TiciVideoStreamTrack):
  kind = "video"
  camera_to_sock_mapping = {
    "driver": "livestreamDriverEncodeData",
    "wideRoad": "livestreamWideRoadEncodeData",
    "road": "livestreamRoadEncodeData",
  }

  def __init__(self, camera_type):
    assert camera_type in self.camera_to_sock_mapping
    super().__init__(camera_type)

    self.sock = messaging.sub_sock(self.camera_to_sock_mapping[camera_type], conflate=True)
    self.dt = DT_MDL
    self.pts = 0

  async def recv(self):
    while True:
      msg = messaging.recv_one_or_none(self.sock)
      if msg is not None:
        break
      await asyncio.sleep(0.005)

    evta = getattr(msg, msg.which())
    self.last_idx = evta.idx.encodeId

    packet = av.Packet(evta.header + evta.data)
    packet.time_base = aiortc.mediastreams.VIDEO_TIME_BASE
    packet.pts = self.pts
    self.pts += self.dt * aiortc.mediastreams.VIDEO_CLOCK_RATE
    return packet


class FrameReaderVideoStreamTrack(TiciVideoStreamTrack):
  kind = "video"

  def __init__(self, input_file, dt=DT_MDL, camera_type="driver"):
    assert camera_type in ["driver", "wideRoad", "road"]
    super().__init__(camera_type)

    #frame_reader = FrameReader(input_file)
    #self.frames = [frame_reader.get(i, pix_fmt="rgb24") for i in range(frame_reader.frame_count)]
    self.frames = [np.zeros((1280, 720, 3), dtype=np.uint8) for i in range(1200)]
    self.frame_count = len(self.frames)
    self.frame_index = 0
    self.dt = dt
    self.pts = 0

  async def recv(self):
    print("-- sending frame: ", self.frame_index)
    img = self.frames[self.frame_index]

    new_frame = av.VideoFrame.from_ndarray(img, format="rgb24")
    new_frame.pts = self.pts
    new_frame.time_base = aiortc.mediastreams.VIDEO_TIME_BASE

    self.frame_index = (self.frame_index + 1) % self.frame_count
    self.pts += self.dt * aiortc.mediastreams.VIDEO_CLOCK_RATE

    return new_frame


class WebRTCStream:
  def __init__(self, sdb, type, video_tracks=[], audio_tracks=[]):
    self.offer = aiortc.RTCSessionDescription(sdp=sdb, type=type)
    self.peer_connection = aiortc.RTCPeerConnection()
    self.peer_connection.on("datachannel", self._on_datachannel)
    self.peer_connection.on("connectionstatechange", self._on_connectionstatechange)

    for video_track in video_tracks:
      video_sender = self.peer_connection.addTrack(video_track)
      self.force_codec(video_sender, "video/H264", "video")
    for audio_track in audio_tracks:
      self.peer_connection.addTrack(audio_track)

  def _on_connectionstatechange(self):
    print("-- connection state is", self.peer_connection.connectionState)

  def _on_datachannel(self, channel):
    pass

  def force_codec(self, sender, codec, stream_type):
    codecs = aiortc.RTCRtpSender.getCapabilities(stream_type).codecs
    codec = [codec for codec in codecs if codec.mimeType == codec]
    transceiver = next(t for t in self.peer_connection.getTransceivers() if t.sender == sender)
    transceiver.setCodecPreferences(codec)

  def start(self):
    answer = asyncio.run(self.start_async())
    return answer

  async def start_async(self):
    await self.peer_connection.setRemoteDescription(self.offer)
    answer = await self.peer_connection.createAnswer()
    await self.peer_connection.setLocalDescription(answer)
    actual_answer = self.peer_connection.localDescription

    return actual_answer


if __name__=="__main__":
  parser = argparse.ArgumentParser(description="WebRTC server")
  parser.add_argument("--input-video", type=str, required=False, help="Stream from video file instead")
  args = parser.parse_args()

  async def async_input():
    return await asyncio.to_thread(input)

  async def run(args):
    streams = []
    while True:
      print("-- Please enter a JSON from client --")
      raw_payload = await async_input()
      
      payload = json.loads(raw_payload)
      offer = WebRTCStreamingOffer(**payload["offer"])
      metadata = WebRTCStreamingMetadata(**payload["metadata"])
      track_gen = lambda cam: FrameReaderVideoStreamTrack(args.input_video, camera_type=cam) if args.input_video else LiveStreamVideoStreamTrack(cam)
      video_tracks = [track_gen(cam) for cam in metadata.video]
      audio_tracks = []

      stream = WebRTCStream(offer.sdp, offer.type, video_tracks, audio_tracks)
      answer = await stream.start_async()
      streams.append(stream)

      print("-- Please send this JSON to client --")
      print(json.dumps({"sdp": answer.sdp, "type": answer.type}))

      await asyncio.sleep(120)
  
  loop = asyncio.get_event_loop()
  loop.run_until_complete(run(args))
