import asyncio
import aiortc
from aiortc.contrib.media import MediaRelay

import abc
import argparse
import json
import logging
from typing import Callable, Awaitable, Dict, List, Any, Optional

from openpilot.tools.bodyteleop.webrtc.common import StreamingOffer, ConnectionProvider


class WebRTCStreamBuilder:
  def __init__(self):
    self.consumed_camera_tracks = set()
    self.consume_audio = False
    self.video_producing_tracks = []
    self.audio_producing_tracks = []
    self.data_channel = False
    self.peer_connection = None

  def add_video_consumer(self, camera_type):
    assert camera_type in ["driver", "wideRoad", "road"]

    self.consumed_camera_tracks.add(camera_type)

  def add_audio_consumer(self):
    self.consume_audio = True

  def add_video_producer(self, track: aiortc.MediaStreamTrack):
    assert track.kind == "video"
    self.video_producing_tracks.append(track)

  def add_audio_producer(self, track: aiortc.MediaStreamTrack):
    assert track.kind == "audio"
    self.audio_producing_tracks.append(track)

  def add_messaging(self):
    self.data_channel = True

  def offer(self, connection_provider: ConnectionProvider):
    return WebRTCOfferStream(
      connection_provider, 
      self.consumed_camera_tracks, 
      self.consume_audio,
      self.data_channel, 
      self.video_producing_tracks,
      self.audio_producing_tracks
    )

  def answer(self, session: aiortc.RTCSessionDescription):
    return WebRTCAnswerStream(
      session, 
      self.consumed_camera_tracks, 
      self.consume_audio, 
      self.data_channel, 
      self.video_producing_tracks, 
      self.audio_producing_tracks
    )


class WebRTCBaseStream(abc.ABC):
  def __init__(self,
               consumed_camera_types: List[str],
               consume_audio: bool,
               enable_messaging: bool,
               video_producer_tracks: List[aiortc.MediaStreamTrack],
               audio_producer_tracks: List[aiortc.MediaStreamTrack]):
    self.peer_connection = aiortc.RTCPeerConnection()
    self.media_relay = MediaRelay()
    self.expected_incoming_camera_types = consumed_camera_types
    self.expected_incoming_audio = consume_audio
    self.expected_number_tracks_or_channels = None

    self.incoming_camera_tracks = dict()
    self.incoming_audio_tracks = []
    self.outgoing_video_tracks = video_producer_tracks
    self.outgoing_audio_tracks = audio_producer_tracks

    self.enable_messaging = enable_messaging
    self.messaging_channel = None
    self.messaging_channel_is_incoming = False
    self.incoming_message_handlers = []

    self.incoming_media_ready_event = asyncio.Event()
    self.connection_attempted_event = asyncio.Event()

    self.peer_connection.on("connectionstatechange", self._on_connectionstatechange)
    self.peer_connection.on("datachannel", self._on_incoming_datachannel)
    self.peer_connection.on("track", self._on_incoming_track)

    self.logger = logging.getLogger("WebRTCStream")
  
  def _log_debug(self, msg: Any, *args):
    self.logger.debug(f"{type(self)}() {msg}", *args)

  @property
  def _number_of_incoming_media(self):
    return len(self.incoming_camera_tracks) + len(self.incoming_audio_tracks) + int(self.messaging_channel_is_incoming)

  def _add_consumer_tracks(self):
    for _ in self.expected_incoming_camera_types:
      self.peer_connection.addTransceiver("video", direction="recvonly")
    if self.expected_incoming_audio:
      self.peer_connection.addTransceiver("audio", direction="recvonly")
  
  def _add_producer_tracks(self):
    for track in self.outgoing_video_tracks:
      sender = self.peer_connection.addTrack(track)
      if hasattr(track, "codec_preference") and track.codec_preference() is not None:
        transceiver = next(t for t in self.peer_connection.getTransceivers() if t.sender == sender)
        codec_mime = f"video/{track.codec_preference().upper()}"
        self._force_codec(transceiver, codec_mime, "video")
    for track in self.outgoing_audio_tracks:
      self.peer_connection.addTrack(track)

  def _add_messaging_channel(self, channel: Optional[aiortc.RTCDataChannel] = None):
    if not channel:
      channel = self.peer_connection.createDataChannel("data", ordered=True)
      self.messaging_channel_is_incoming = False
    else:
      self.messaging_channel_is_incoming = True

    for handler in self.incoming_message_handlers:
      channel.on("message", self._create_channel_handler_wrapper(channel, handler))
    self.messaging_channel = channel

  def _create_channel_handler_wrapper(self, channel, message_handler):
    async def handler_wrapper(message):
      await message_handler(channel, message)
    return handler_wrapper
  
  def _force_codec(self, transceiver, codec_mime, stream_type):
    codecs = aiortc.RTCRtpSender.getCapabilities(stream_type).codecs
    codec = [codec for codec in codecs if codec.mimeType == codec_mime]
    transceiver.setCodecPreferences(codec)

  def _on_connectionstatechange(self):
    self._log_debug("connection state is", self.peer_connection.connectionState)
    if self.peer_connection.connectionState in ['connected', 'failed']:
      self.connection_attempted_event.set()

  def _on_incoming_track(self, track: aiortc.MediaStreamTrack):
    self._log_debug("got track: ", track.kind, track.id)
    if track.kind == "video":
      parts = track.id.split(":") # format: "camera_type:camera_id"
      if len(parts) < 2:
        return

      camera_type = parts[0]
      if camera_type in self.expected_incoming_camera_types:
        self.incoming_camera_tracks[camera_type] = track
    elif track.kind == "audio":
      if self.expected_incoming_audio:
        self.incoming_audio_tracks.append(track)
    self._on_after_media()

  def _on_incoming_datachannel(self, channel: aiortc.RTCDataChannel):
    self._log_debug("got data channel: ", channel.label)
    if channel.label == "data" and self.messaging_channel is None:
      self._add_messaging_channel(channel)
    self._on_after_media()

  def _on_after_media(self):
    if self._number_of_incoming_media == self.expected_number_tracks_or_channels:
      self.incoming_media_ready_event.set()

  def _parse_incoming_streams(self, remote_sdp: str):
    desc = aiortc.sdp.SessionDescription.parse(remote_sdp)
    sending_medias = [m for m in desc.media if m.direction in ["sendonly", "sendrecv"]]
    self.expected_number_tracks_or_channels = len(sending_medias)

  def has_incoming_video_track(self, camera_type: str) -> bool:
    return camera_type in self.incoming_camera_tracks

  def has_incoming_audio_track(self) -> bool:
    return len(self.incoming_audio_tracks) > 0

  def get_incoming_video_track(self, camera_type: str, buffered: bool):
    assert camera_type in self.incoming_camera_tracks, "Video tracks are not enabled on this stream"
    assert self.peer_connection is not None, "Stream must be started"

    track = self.incoming_camera_tracks[camera_type]
    relay_track = self.media_relay.subscribe(track, buffered=buffered)
    return relay_track
  
  def get_incoming_audio_track(self, buffered: bool):
    assert len(self.incoming_audio_tracks) > 0, "Audio tracks are not enabled on this stream"
    assert self.peer_connection is not None, "Stream must be started"

    track = self.incoming_audio_tracks[0]
    relay_track = self.media_relay.subscribe(track, buffered=buffered)
    return relay_track
  
  def get_messaging_channel(self):
    assert self.enable_messaging, "Messaging is not enabled on this stream"
    assert self.peer_connection is not None, "Stream must be started"

    return self.messaging_channel

  def set_message_handler(self, message_handler: Callable[[aiortc.RTCDataChannel, bytes], Awaitable[None]]):
    self.incoming_message_handlers.append(message_handler)
    if self.is_started:
      self.messaging_channel.on("message", self._create_channel_handler_wrapper(self.messaging_channel, message_handler))

  @property
  def is_started(self) -> bool:
    return self.peer_connection is not None and \
           self.peer_connection.localDescription is not None and \
           self.peer_connection.remoteDescription is not None

  async def wait_for_connection(self):
    await self.connection_attempted_event.wait()
    if self.peer_connection.connectionState != 'connected':
      raise ValueError("Connection failed.")
    if self.expected_number_tracks_or_channels:
      await self.incoming_media_ready_event.wait()
  
  async def stop(self):
    await self.peer_connection.close()

  @abc.abstractmethod
  async def start(self):
    raise NotImplemented


class WebRTCOfferStream(WebRTCBaseStream):
  def __init__(self, session_provider: ConnectionProvider, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.session_provider = session_provider

  async def start(self):
    self._add_consumer_tracks()
    if self.enable_messaging:
      self._add_messaging_channel()
    self._add_producer_tracks()

    offer = await self.peer_connection.createOffer()
    await self.peer_connection.setLocalDescription(offer)
    actual_offer = self.peer_connection.localDescription

    streaming_offer = StreamingOffer(
      sdp=actual_offer.sdp, 
      type=actual_offer.type, 
      video=list(self.expected_incoming_camera_types), 
      audio=self.expected_incoming_audio,
      channel=self.enable_messaging
    )
    remote_answer = await self.session_provider(streaming_offer)
    self._parse_incoming_streams(remote_sdp=remote_answer.sdp)
    await self.peer_connection.setRemoteDescription(remote_answer)
    actual_answer = self.peer_connection.remoteDescription

    return actual_answer


class WebRTCAnswerStream(WebRTCBaseStream):
  def __init__(self, session: aiortc.RTCSessionDescription, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.session = session

  def _probe_video_codecs(self) -> List[str]:
    codecs = []
    for track in self.outgoing_video_tracks:
      if hasattr(track, "codec_preference") and track.codec_preference() is not None:
        codecs.append(track.codec_preference())

    return codecs

  def _override_incoming_video_codecs(self, remote_sdp: str, preferred_codecs: List[str]) -> str:
    desc = aiortc.sdp.SessionDescription.parse(remote_sdp)
    preferred_codec_mimes = [f"video/{c}" for c in preferred_codecs]
    for m in desc.media:
      if m.kind != "video":
        continue


      preferred_codecs = [c for c in m.rtp.codecs if c.mimeType in preferred_codec_mimes]
      if len(preferred_codecs) == 0:
        raise ValueError(f"None of {preferred_codecs} codecs is supported in remote SDP")

      m.rtp.codecs = preferred_codecs
      m.fmt = [c.payloadType for c in preferred_codecs]

    return str(desc)
  
  def _assert_incoming_sdp(self, sdp: str):
    def n_media_senders(kind, medias):
      return [m for m in medias if m.kind == kind and m.direction in ["sendonly", "sendrecv"]]
    def n_media_receivers(kind, medias):
      return [m for m in medias if m.kind == kind and m.direction in ["recvonly", "sendrecv"]]
    desc = aiortc.sdp.SessionDescription.parse(sdp)
    transceivers = self.peer_connection.getTransceivers()
    
    enough_remote_video_recv = n_media_receivers("video", desc.media) >= n_media_senders("video", transceivers)
    enough_remote_audio_recv = n_media_receivers("audio", desc.media) >= n_media_senders("audio", transceivers)
    enough_local_video_recv = n_media_receivers("video", transceivers) >= n_media_senders("video", desc.media)
    enough_local_audio_recv = n_media_receivers("audio", transceivers) >= n_media_senders("audio", desc.media) 

    assert enough_remote_video_recv, "Not enough video receivers on remote peer"
    assert enough_remote_audio_recv, "Not enough audio receivers on remote peer"
    assert enough_local_video_recv, "Not enough video receivers on local peer"
    assert enough_local_audio_recv, "Not enough audio receivers on local peer"

  async def start(self):
    assert self.peer_connection.remoteDescription is None, "Connection already established"

    # since we sent already encoded frames in some cases (e.g. livestream video tracks are in H264), we need to force aiortc to actually use it
    # we do that by overriding supported codec information on incoming sdp
    preferred_codecs = self._probe_video_codecs()
    if len(preferred_codecs) > 0:
      self.session.sdp = self._override_incoming_video_codecs(self.session.sdp, preferred_codecs)

    self._parse_incoming_streams(remote_sdp=self.session.sdp)
    await self.peer_connection.setRemoteDescription(self.session)

    self._add_consumer_tracks()
    if self.enable_messaging:
      self._add_messaging_channel()
    self._add_producer_tracks()

    answer = await self.peer_connection.createAnswer()
    await self.peer_connection.setLocalDescription(answer)
    actual_answer = self.peer_connection.localDescription

    return actual_answer


if __name__ == "__main__":
  from openpilot.tools.bodyteleop.webrtc.tracks import LiveStreamVideoStreamTrack, FrameReaderVideoStreamTrack, DummyVideoStreamTrack
  from openpilot.tools.bodyteleop.webrtc.common import StdioConnectionProvider

  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(dest="command", required=True)
  offer_parser = subparsers.add_parser("offer")
  offer_parser.add_argument("cameras", metavar="CAMERA", type=str, nargs="+", default=["driver"], help="Camera types to stream")

  answer_parser = subparsers.add_parser("answer")
  answer_parser.add_argument("--dummy-video", action="store_true", help="Stream dummy frames")
  answer_parser.add_argument("--input-video", type=str, required=False, help="Stream from video file instead")
  
  args = parser.parse_args()

  async def async_input():
    return await asyncio.to_thread(input)

  async def run_answer(args):
    streams = []
    while True:
      print("-- Please enter a JSON from client --")
      raw_payload = await async_input()
      
      payload = json.loads(raw_payload)
      offer = StreamingOffer(**payload)
      video_tracks = []
      for cam in offer.video:
        if args.dummy_video:
          track = DummyVideoStreamTrack(camera_type=cam)
        elif args.input_video:
          track = FrameReaderVideoStreamTrack(args.input_video, camera_type=cam)
        else:
          track = LiveStreamVideoStreamTrack(cam)
        video_tracks.append(track)
      audio_tracks = []

      stream_builder = WebRTCStreamBuilder()
      for track in video_tracks:
        stream_builder.add_video_producer(track)
      for track in audio_tracks:
        stream_builder.add_audio_producer(track)
      stream = stream_builder.answer(offer)
      answer = await stream.start()
      streams.append(stream)

      print("-- Please send this JSON to client --")
      print(json.dumps({"sdp": answer.sdp, "type": answer.type}))

      await stream.wait_for_connection()

  async def run_offer(args):
    connection_provider = StdioConnectionProvider()
    stream_builder = WebRTCStreamBuilder()
    for cam in args.cameras:
      stream_builder.add_video_consumer(cam)
    stream_builder.add_messaging()
    stream = stream_builder.offer(connection_provider)
    _ = await stream.start()
    await stream.wait_for_connection()

    tracks = [stream.get_incoming_video_track(cam, False) for cam in args.cameras]
    while True:
      try:
        frames = await asyncio.gather(*[track.recv() for track in tracks])
        for key, frame in zip(args.cameras, frames):
          print("Received frame from", key, frame.time)
      except aiortc.mediastreams.MediaStreamError:
        return
      print("=====================================")

  loop = asyncio.get_event_loop()
  if args.command == "offer":
    loop.run_until_complete(run_offer(args))
  elif args.command == "answer":
    loop.run_until_complete(run_answer(args))
