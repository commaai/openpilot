import abc
import asyncio
import dataclasses
import logging
from typing import Any
from collections.abc import Callable, Awaitable

import aiortc
from aiortc.contrib.media import MediaRelay

from teleoprtc.tracks import parse_video_track_id


@dataclasses.dataclass
class StreamingOffer:
  sdp: str
  video: list[str]


ConnectionProvider = Callable[[StreamingOffer], Awaitable[aiortc.RTCSessionDescription]]
MessageHandler = Callable[[bytes], Awaitable[None]]


class WebRTCBaseStream(abc.ABC):
  def __init__(self,
               consumed_camera_types: list[str],
               consume_audio: bool,
               video_producer_tracks: list[aiortc.MediaStreamTrack],
               audio_producer_tracks: list[aiortc.MediaStreamTrack],
               should_add_data_channel: bool):
    self.peer_connection = aiortc.RTCPeerConnection()
    self.media_relay = MediaRelay()
    self.expected_incoming_camera_types = consumed_camera_types
    self.expected_incoming_audio = consume_audio
    self.expected_number_of_incoming_media: int | None = None

    self.incoming_camera_tracks: dict[str, aiortc.MediaStreamTrack] = dict()
    self.incoming_audio_tracks: list[aiortc.MediaStreamTrack] = []
    self.outgoing_video_tracks: list[aiortc.MediaStreamTrack] = video_producer_tracks
    self.outgoing_audio_tracks: list[aiortc.MediaStreamTrack] = audio_producer_tracks

    self.should_add_data_channel = should_add_data_channel
    self.messaging_channel: aiortc.RTCDataChannel | None = None
    self.incoming_message_handlers: list[MessageHandler] = []

    self.incoming_media_ready_event = asyncio.Event()
    self.messaging_channel_ready_event = asyncio.Event()
    self.connection_attempted_event = asyncio.Event()
    self.connection_stopped_event = asyncio.Event()

    self.peer_connection.on("connectionstatechange", self._on_connectionstatechange)
    self.peer_connection.on("datachannel", self._on_incoming_datachannel)
    self.peer_connection.on("track", self._on_incoming_track)

    self.logger = logging.getLogger("WebRTCStream")

  def _log_debug(self, msg: Any, *args):
    self.logger.debug(f"{type(self)}() {msg}", *args)

  @property
  def _number_of_incoming_media(self) -> int:
    media = len(self.incoming_camera_tracks) + len(self.incoming_audio_tracks)
    # if stream does not add data_channel, then it means its incoming
    media += int(self.messaging_channel is not None) if not self.should_add_data_channel else 0
    return media

  def _add_consumer_transceivers(self):
    for _ in self.expected_incoming_camera_types:
      self.peer_connection.addTransceiver("video", direction="recvonly")
    if self.expected_incoming_audio:
      self.peer_connection.addTransceiver("audio", direction="recvonly")

  def _find_trackless_transceiver(self, kind: str) -> aiortc.RTCRtpTransceiver | None:
    transceivers = self.peer_connection.getTransceivers()
    target_transceiver = None
    for t in transceivers:
      if t.kind == kind and t.sender.track is None:
        target_transceiver = t
        break

    return target_transceiver

  def _add_producer_tracks(self):
    for track in self.outgoing_video_tracks:
      target_transceiver = self._find_trackless_transceiver(track.kind)
      if target_transceiver is None:
        self.peer_connection.addTransceiver(track.kind, direction="sendonly")

      sender = self.peer_connection.addTrack(track)
      if hasattr(track, "codec_preference") and track.codec_preference() is not None:
        transceiver = next(t for t in self.peer_connection.getTransceivers() if t.sender == sender)
        self._force_codec(transceiver, track.codec_preference(), "video")
    for track in self.outgoing_audio_tracks:
      target_transceiver = self._find_trackless_transceiver(track.kind)
      if target_transceiver is None:
        self.peer_connection.addTransceiver(track.kind, direction="sendonly")

      self.peer_connection.addTrack(track)

  def _add_messaging_channel(self, channel: aiortc.RTCDataChannel | None = None):
    if not channel:
      channel = self.peer_connection.createDataChannel("data", ordered=True)

    for handler in self.incoming_message_handlers:
      channel.on("message", handler)

    if channel.readyState == "open":
      self.messaging_channel_ready_event.set()
    else:
      channel.on("open", lambda: self.messaging_channel_ready_event.set())
    self.messaging_channel = channel

  def _force_codec(self, transceiver: aiortc.RTCRtpTransceiver, codec: str, stream_type: str):
    codec_mime = f"{stream_type}/{codec.upper()}"
    rtp_codecs = aiortc.RTCRtpSender.getCapabilities(stream_type).codecs
    rtp_codec = [c for c in rtp_codecs if c.mimeType == codec_mime]
    transceiver.setCodecPreferences(rtp_codec)

  def _on_connectionstatechange(self):
    self._log_debug("connection state is %s", self.peer_connection.connectionState)
    if self.peer_connection.connectionState in ['connected', 'failed']:
      self.connection_attempted_event.set()
    if self.peer_connection.connectionState in ['disconnected', 'closed', 'failed']:
      self.connection_stopped_event.set()

  def _on_incoming_track(self, track: aiortc.MediaStreamTrack):
    self._log_debug("got track: %s %s", track.kind, track.id)
    if track.kind == "video":
      camera_type, _ = parse_video_track_id(track.id)
      if camera_type in self.expected_incoming_camera_types:
        self.incoming_camera_tracks[camera_type] = track
    elif track.kind == "audio":
      if self.expected_incoming_audio:
        self.incoming_audio_tracks.append(track)
    self._on_after_media()

  def _on_incoming_datachannel(self, channel: aiortc.RTCDataChannel):
    self._log_debug("got data channel: %s", channel.label)
    if channel.label == "data" and self.messaging_channel is None:
      self._add_messaging_channel(channel)
    self._on_after_media()

  def _on_after_media(self):
    if self._number_of_incoming_media == self.expected_number_of_incoming_media:
      self.incoming_media_ready_event.set()

  def _parse_incoming_streams(self, remote_sdp: str):
    desc = aiortc.sdp.SessionDescription.parse(remote_sdp)
    sending_medias = [m for m in desc.media if m.direction in ["sendonly", "sendrecv"]]
    incoming_media_count = len(sending_medias)
    if not self.should_add_data_channel:
      channel_medias = [m for m in desc.media if m.kind == "application"]
      incoming_media_count += len(channel_medias)
    self.expected_number_of_incoming_media = incoming_media_count

  def has_incoming_video_track(self, camera_type: str) -> bool:
    return camera_type in self.incoming_camera_tracks

  def has_incoming_audio_track(self) -> bool:
    return len(self.incoming_audio_tracks) > 0

  def has_messaging_channel(self) -> bool:
    return self.messaging_channel is not None

  def get_incoming_video_track(self, camera_type: str, buffered: bool = False) -> aiortc.MediaStreamTrack:
    assert camera_type in self.incoming_camera_tracks, "Video tracks are not enabled on this stream"
    assert self.is_started, "Stream must be started"

    track = self.incoming_camera_tracks[camera_type]
    relay_track = self.media_relay.subscribe(track, buffered=buffered)
    return relay_track

  def get_incoming_audio_track(self, buffered: bool = False) -> aiortc.MediaStreamTrack:
    assert len(self.incoming_audio_tracks) > 0, "Audio tracks are not enabled on this stream"
    assert self.is_started, "Stream must be started"

    track = self.incoming_audio_tracks[0]
    relay_track = self.media_relay.subscribe(track, buffered=buffered)
    return relay_track

  def get_messaging_channel(self) -> aiortc.RTCDataChannel:
    assert self.messaging_channel is not None, "Messaging channel is not enabled on this stream"
    assert self.is_started, "Stream must be started"

    return self.messaging_channel

  def set_message_handler(self, message_handler: MessageHandler):
    self.incoming_message_handlers.append(message_handler)
    if self.messaging_channel is not None:
      self.messaging_channel.on("message", message_handler)

  @property
  def is_started(self) -> bool:
    return self.peer_connection is not None and \
           self.peer_connection.localDescription is not None and \
           self.peer_connection.remoteDescription is not None and \
           self.peer_connection.connectionState != "closed"

  @property
  def is_connected_and_ready(self) -> bool:
    return self.peer_connection is not None and \
           self.peer_connection.connectionState == "connected" and \
           (self.expected_number_of_incoming_media == 0 or self.incoming_media_ready_event.is_set())

  async def wait_for_connection(self):
    assert self.is_started
    await self.connection_attempted_event.wait()
    if self.peer_connection.connectionState != 'connected':
      raise ValueError("Connection failed.")
    if self.expected_number_of_incoming_media:
      await self.incoming_media_ready_event.wait()
    if self.messaging_channel is not None:
      await self.messaging_channel_ready_event.wait()

  async def wait_for_disconnection(self):
    assert self.is_connected_and_ready, "Stream is not connected/ready yet (make sure wait_for_connection was awaited)"
    await self.connection_stopped_event.wait()

  async def stop(self):
    await self.peer_connection.close()

  @abc.abstractmethod
  async def start(self) -> aiortc.RTCSessionDescription:
    raise NotImplementedError


class WebRTCOfferStream(WebRTCBaseStream):
  def __init__(self, session_provider: ConnectionProvider, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.session_provider = session_provider

  async def start(self) -> aiortc.RTCSessionDescription:
    self._add_consumer_transceivers()
    if self.should_add_data_channel:
      self._add_messaging_channel()
    self._add_producer_tracks()

    offer = await self.peer_connection.createOffer()
    await self.peer_connection.setLocalDescription(offer)
    actual_offer = self.peer_connection.localDescription

    streaming_offer = StreamingOffer(
      sdp=actual_offer.sdp,
      video=list(self.expected_incoming_camera_types),
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

  def _probe_video_codecs(self) -> list[str]:
    codecs = []
    for track in self.outgoing_video_tracks:
      if hasattr(track, "codec_preference") and track.codec_preference() is not None:
        codecs.append(track.codec_preference())

    return codecs

  def _override_incoming_video_codecs(self, remote_sdp: str, codecs: list[str]) -> str:
    desc = aiortc.sdp.SessionDescription.parse(remote_sdp)
    codec_mimes = [f"video/{c}" for c in codecs]
    for m in desc.media:
      if m.kind != "video":
        continue

      preferred_codecs: list[aiortc.RTCRtpCodecParameters] = [c for c in m.rtp.codecs if c.mimeType in codec_mimes]
      if len(preferred_codecs) == 0:
        raise ValueError(f"None of {preferred_codecs} codecs is supported in remote SDP")

      m.rtp.codecs = preferred_codecs
      m.fmt = [c.payloadType for c in preferred_codecs]

    return str(desc)

  async def start(self) -> aiortc.RTCSessionDescription:
    assert self.peer_connection.remoteDescription is None, "Connection already established"

    self._add_consumer_transceivers()

    # since we sent already encoded frames in some cases (e.g. livestream video tracks are in H264), we need to force aiortc to actually use it
    # we do that by overriding supported codec information on incoming sdp
    preferred_codecs = self._probe_video_codecs()
    if len(preferred_codecs) > 0:
      self.session.sdp = self._override_incoming_video_codecs(self.session.sdp, preferred_codecs)

    self._parse_incoming_streams(remote_sdp=self.session.sdp)
    await self.peer_connection.setRemoteDescription(self.session)

    self._add_producer_tracks()

    answer = await self.peer_connection.createAnswer()
    await self.peer_connection.setLocalDescription(answer)
    actual_answer = self.peer_connection.localDescription

    return actual_answer
