#!/usr/bin/env python3

import argparse
import json
from typing import Awaitable, Callable, Any

import aiortc
import av
import asyncio
import numpy as np

from cereal import messaging
from openpilot.common.realtime import DT_MDL, DT_DMON
from openpilot.tools.bodyteleop.webrtc.client import DataChannelMessenger
from openpilot.tools.bodyteleop.webrtc.connection import StreamingOffer
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
    shape = (1280, 720, 3) if camera_type == "driver" else (1920, 1080, 3)
    self.frames = [np.zeros(shape, dtype=np.uint8) for i in range(1200)]
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


import pyaudio
from pydub import AudioSegment


class AudioInputStreamTrack(aiortc.mediastreams.AudioStreamTrack):
  def __init__(self, format=pyaudio.paInt16, rate=16000, channels=1, packet_time=0.020, device_index=None):
    super().__init__()

    self.p = pyaudio.PyAudio()
    frame_per_buffer = packet_time * rate
    self.stream = self.p.open(format=format, 
                              channels=channels, 
                              rate=rate, 
                              frames_per_buffer=frame_per_buffer, 
                              input=True, 
                              input_device_index=device_index)
    self.format = format
    self.rate = rate
    self.channels = channels
    self.packet_time = packet_time
    self.chunk_size = int(rate * packet_time)
    self.chunk_index = 0
    self.pts = 0

  async def recv(self):
    mic_data = self.stream.read(self.chunk_size)
    mic_sound = AudioSegment(mic_data, sample_width=pyaudio.get_sample_size(self.format), channels=self.channels, frame_rate=self.rate)
    # create stereo sound?
    mic_sound = AudioSegment.from_mono_audiosegments(mic_sound, mic_sound)
    packet = av.Packet(mic_sound.raw_data)
    # TODO
    return None


class WebRTCStream:
  def __init__(self, sdb, type, video_codec="video/H264", audio_codec=None):
    self.offer = aiortc.RTCSessionDescription(sdp=sdb, type=type)
    self.peer_connection = aiortc.RTCPeerConnection()
    self.peer_connection.on("datachannel", self._on_datachannel)
    self.peer_connection.on("connectionstatechange", self._on_connectionstatechange)
    self.video_tracks = []
    self.video_codec = video_codec
    self.audio_tracks = []
    self.audio_codec = audio_codec
    self.message_handler = None

  def _on_connectionstatechange(self):
    print("-- connection state is", self.peer_connection.connectionState)

  def _on_datachannel(self, channel):
    async def on_message(message):
      if self.message_handler:
        await self.message_handler(DataChannelMessenger(channel), message)

    channel.on("message", on_message)

  def _force_codec(self, sender, codec, stream_type):
    codecs = aiortc.RTCRtpSender.getCapabilities(stream_type).codecs
    codec = [codec for codec in codecs if codec.mimeType == codec]
    transceiver = next(t for t in self.peer_connection.getTransceivers() if t.sender == sender)
    transceiver.setCodecPreferences(codec)

  def add_video_track(self, track: aiortc.MediaStreamTrack):
    assert track.kind == "video"
    self.video_tracks.append(track)

  def add_audio_track(self, track: aiortc.MediaStreamTrack):
    assert track.kind == "audio"
    self.audio_tracks.append(track)
  
  def add_message_handler(self, handler: Callable[[DataChannelMessenger, Any], Awaitable[None]]):
    self.message_handler = handler

  async def start_async(self):
    assert self.peer_connection.remoteDescription is None, "Connection already established"

    await self.peer_connection.setRemoteDescription(self.offer)

    for video_track in self.video_tracks:
      video_sender = self.peer_connection.addTrack(video_track)
      if self.video_codec:
        self._force_codec(video_sender, self.video_codec, "video")
    for audio_track in self.audio_tracks:
      audio_sender = self.peer_connection.addTrack(audio_track)
      if self.audio_codec:
        self._force_codec(audio_sender, self.audio_codec, "audio")

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
      offer = StreamingOffer(**payload)
      video_tracks = []
      for cam in offer.video:
        if args.input_video:
          track = FrameReaderVideoStreamTrack(args.input_video, camera_type=cam)
        else:
          track = LiveStreamVideoStreamTrack(cam)
        video_tracks.append(track)
      audio_tracks = []

      stream = WebRTCStream(offer.sdp, offer.type)
      for track in video_tracks:
        stream.add_video_track(track)
      for track in audio_tracks:
        stream.add_audio_track(track)
      answer = await stream.start_async()
      streams.append(stream)

      print("-- Please send this JSON to client --")
      print(json.dumps({"sdp": answer.sdp, "type": answer.type}))
  
  loop = asyncio.get_event_loop()
  loop.run_until_complete(run(args))
