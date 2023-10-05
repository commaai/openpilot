import aiortc
from aiortc.mediastreams import VIDEO_CLOCK_RATE, VIDEO_TIME_BASE
import av
import asyncio

from cereal import messaging
from openpilot.common.realtime import DT_MDL, DT_DMON


class LiveStreamVideoStreamTrack(aiortc.MediaStreamTrack):
  kind = "video"

  def __init__(self, sock_name):
    super().__init__()

    self.sock = messaging.sub_sock(sock_name, conflate=True)
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


class FrameReaderVideoStreamTrack(aiortc.MediaStreamTrack):
  kind = "video"

  def __init__(self, frame_reader, dt=DT_MDL):
    super().__init__()
    self.frames = [frame_reader.get(i, pix_fmt="rgb24") for i in frame_reader.frame_count]
    self.frame_count = len(self.frames)
    self.frame_index = 0
    self.dt = dt
    self.pts = 0

  async def recv(self):
    img = self.frames[self.frame_index]

    new_frame = av.VideoFrame.from_ndarray(img, format="rgb24")
    new_frame.pts = self.pts
    new_frame.time_base = aiortc.mediastreams.VIDEO_TIME_BASE

    self.frame_index = (self.frame_index + 1) % self.frame_count
    self.pts += self.dt * aiortc.mediastreams.VIDEO_CLOCK_RATE

    return new_frame


class WebRTCStream:
  def __init__(self, sdb, type, video_track=None, audio_track=None):
    self.offer = aiortc.RTCSessionDescription(sdp=sdb, type=type)
    self.peer_connection = aiortc.RTCPeerConnection()

    if video_track is not None:
      video_sender = self.peer_connection.addTrack(video_track)
      self.force_codec(video_sender, "video/H264", "video")
    if audio_track is not None:
      self.peer_connection.addTrack(audio_track)

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

    return answer
