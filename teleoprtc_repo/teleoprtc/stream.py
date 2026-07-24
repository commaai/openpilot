import abc
import asyncio
import contextlib
import dataclasses
import logging
import random
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

from libdatachannel import (
  Configuration,
  DataChannel,
  Description,
  FrameInfo,
  H264RtpPacketizer,
  IceServer,
  NalUnit,
  PeerConnection,
  PliHandler,
  RtcpNackResponder,
  RtcpSrReporter,
  RtpPacketizationConfig,
  Track,
)

from teleoprtc.decoder import RtcpReceiverReport, _decode_receiver_reports
from teleoprtc.tracks import TiciVideoStreamTrack, parse_video_track_id


@dataclasses.dataclass
class StreamingOffer:
  sdp: str
  video: List[str]


@dataclasses.dataclass
class RTCSessionDescription:
  sdp: str
  type: str


ConnectionProvider = Callable[[StreamingOffer], Awaitable[RTCSessionDescription]]
MessageHandler = Callable[[Union[bytes, str]], None]


class WebRTCBaseStream(abc.ABC):
  def __init__(self,
               consumed_camera_types: List[str],
               consume_audio: bool,
               video_producer_tracks: List[TiciVideoStreamTrack],
               audio_producer_tracks: List[Any],
               should_add_data_channel: bool,
               bind_address: Optional[str] = None):
    config = Configuration()
    config.force_media_transport = True
    config.disable_auto_negotiation = True
    config.ice_servers = [IceServer("stun:stun.l.google.com:19302")]
    if bind_address is not None:
      config.bind_address = bind_address

    self.peer_connection = PeerConnection(config)
    self.expected_incoming_camera_types = consumed_camera_types
    self.expected_incoming_audio = consume_audio
    self.expected_number_of_incoming_media: Optional[int] = None

    self.incoming_camera_tracks: Dict[str, Any] = {}
    self.incoming_audio_tracks: List[Any] = []
    self.outgoing_video_tracks = video_producer_tracks
    self.outgoing_audio_tracks = audio_producer_tracks

    self.should_add_data_channel = should_add_data_channel
    self.messaging_channel: Optional[DataChannel] = None
    self.incoming_message_handlers: List[MessageHandler] = []
    self._consumer_tracks: List[Track] = []
    self._sender_tasks: List[asyncio.Task] = []
    self._track_state: List[Tuple[Track, TiciVideoStreamTrack, RtpPacketizationConfig]] = []
    self._receiver_reports: Dict[str, RtcpReceiverReport] = {}
    self._receiver_report_tracks: Dict[str, Tuple[Track, int]] = {}

    self.incoming_media_ready_event = asyncio.Event()
    self.messaging_channel_ready_event = asyncio.Event()
    self.connection_attempted_event = asyncio.Event()
    self.connection_stopped_event = asyncio.Event()
    self.gathering_complete_event = asyncio.Event()
    self._loop: Optional[asyncio.AbstractEventLoop] = None

    self.peer_connection.on_state_change(self._on_connectionstatechange)
    self.peer_connection.on_gathering_state_change(self._on_gatheringstatechange)
    self.peer_connection.on_data_channel(self._on_incoming_datachannel)
    if self.expected_incoming_camera_types or self.expected_incoming_audio:
      self.peer_connection.on_track(self._on_incoming_track)

    self.logger = logging.getLogger("WebRTCStream")

  def _log_debug(self, msg: Any, *args):
    self.logger.debug(f"{type(self)}() {msg}", *args)

  def _call_soon_threadsafe(self, fn: Callable, *args) -> None:
    if self._loop is not None and self._loop.is_running():
      self._loop.call_soon_threadsafe(fn, *args)
    else:
      fn(*args)

  def _set_event(self, event: asyncio.Event) -> None:
    self._call_soon_threadsafe(event.set)

  @property
  def _number_of_incoming_media(self) -> int:
    media = len(self.incoming_camera_tracks) + len(self.incoming_audio_tracks)
    media += int(self.messaging_channel is not None) if not self.should_add_data_channel else 0
    return media

  def _add_consumer_transceivers(self):
    for camera_type in self.expected_incoming_camera_types:
      media = Description.Video(camera_type, Description.Direction.RecvOnly)
      media.add_h264_codec(96)
      track = self.peer_connection.add_track(media)
      self._consumer_tracks.append(track)
      self.incoming_camera_tracks[camera_type] = track
    if self.expected_incoming_audio:
      media = Description.Audio("audio", Description.Direction.RecvOnly)
      media.add_opus_codec(111)
      track = self.peer_connection.add_track(media)
      self._consumer_tracks.append(track)
      self.incoming_audio_tracks.append(track)

  def _find_offer_video(self, remote_sdp: str) -> Tuple[str, int]:
    desc = Description(remote_sdp, Description.Type.Offer)
    for i in range(desc.media_count()):
      media = desc.media(i)
      if media is None or media.type() != "video":
        continue
      for payload_type in media.payload_types():
        with contextlib.suppress(ValueError):
          rtp_map = media.rtp_map(payload_type)
          if rtp_map is not None and rtp_map.format.upper() == "H264":
            return media.mid(), payload_type
    raise ValueError("Remote SDP does not offer H264 video")

  def _make_video_media(self, track: TiciVideoStreamTrack, remote_sdp: str) -> Tuple[Description.Video, int, int, str]:
    mid, payload_type = self._find_offer_video(remote_sdp)
    ssrc = random.randint(1, 0xFFFFFFFF)
    cname = f"teleoprtc-{random.getrandbits(32):08x}"
    stream_id = f"stream-{random.getrandbits(32):08x}"
    media = Description.Video(mid, Description.Direction.SendOnly)
    media.add_h264_codec(payload_type)
    media.add_ssrc(ssrc, cname, stream_id, track.id)
    return media, ssrc, payload_type, cname

  def _add_producer_tracks(self, remote_sdp: Optional[str] = None):
    for track in self.outgoing_video_tracks:
      media, ssrc, payload_type, cname = self._make_video_media(track, remote_sdp or "")
      rtc_track = self.peer_connection.add_track(media)

      rtp_config = RtpPacketizationConfig(ssrc, cname, payload_type, H264RtpPacketizer.CLOCK_RATE)
      rtp_config.start_timestamp = random.randint(0, 0xFFFFFFFF)
      rtp_config.timestamp = rtp_config.start_timestamp
      rtp_config.sequence_number = random.randint(0, 0xFFFF)

      packetizer = H264RtpPacketizer(NalUnit.Separator.LongStartSequence, rtp_config, 1200)
      packetizer.add_to_chain(RtcpSrReporter(rtp_config))
      packetizer.add_to_chain(PliHandler(track.request_keyframe))
      packetizer.add_to_chain(RtcpNackResponder())
      rtc_track.set_media_handler(packetizer)

      camera_type, _ = parse_video_track_id(track.id)
      rtc_track.reset_callbacks()
      self._receiver_report_tracks[camera_type] = (rtc_track, ssrc)
      self._track_state.append((rtc_track, track, rtp_config))

    if self.outgoing_audio_tracks:
      raise NotImplementedError("Audio producer tracks are not implemented with libdatachannel")

  def _add_messaging_channel(self, channel: Optional[DataChannel] = None):
    if channel is None:
      channel = self.peer_connection.create_data_channel("data")
    self.messaging_channel = channel

    def on_message(message: Union[bytes, str]):
      for handler in list(self.incoming_message_handlers):
        self._call_soon_threadsafe(handler, message)

    channel.on_message(on_message)
    channel.on_open(lambda: self._set_event(self.messaging_channel_ready_event))
    channel.on_closed(lambda: self._set_event(self.connection_stopped_event))
    if channel.is_open():
      self._set_event(self.messaging_channel_ready_event)
    self._on_after_media()

  def _on_connectionstatechange(self, state: PeerConnection.State):
    self._log_debug("connection state is %s", state)
    if state in (PeerConnection.State.Connected, PeerConnection.State.Failed):
      self._set_event(self.connection_attempted_event)
    if state in (PeerConnection.State.Disconnected, PeerConnection.State.Closed, PeerConnection.State.Failed):
      self._set_event(self.connection_stopped_event)

  def _on_gatheringstatechange(self, state: PeerConnection.GatheringState):
    self._log_debug("gathering state is %s", state)
    if state == PeerConnection.GatheringState.Complete:
      self._set_event(self.gathering_complete_event)

  def _on_incoming_track(self, track: Track):
    self._log_debug("got track: %s", track.mid())
    try:
      camera_type, _ = parse_video_track_id(track.mid())
    except ValueError:
      camera_type = track.mid()
    if camera_type in self.expected_incoming_camera_types:
      self.incoming_camera_tracks[camera_type] = track
    elif self.expected_incoming_audio:
      self.incoming_audio_tracks.append(track)
    self._on_after_media()

  def _on_incoming_datachannel(self, channel: DataChannel):
    self._log_debug("got data channel: %s", channel.label())
    if channel.label() == "data" and self.messaging_channel is None:
      self._add_messaging_channel(channel)

  def _update_receiver_report(self, camera_type: str, ssrc: int, message: bytes) -> None:
    for report in _decode_receiver_reports(message):
      if report.ssrc == ssrc:
        self._receiver_reports[camera_type] = report

  def _on_after_media(self):
    if self.expected_number_of_incoming_media is not None and self._number_of_incoming_media >= self.expected_number_of_incoming_media:
      self._set_event(self.incoming_media_ready_event)

  def _parse_incoming_streams(self, remote_sdp: str):
    desc = Description(remote_sdp, Description.Type.Offer)
    media_count = 0
    for i in range(desc.media_count()):
      media = desc.media(i)
      if media is None:
        continue
      direction = media.direction()
      if media.type() in ("audio", "video") and direction in (Description.Direction.SendOnly, Description.Direction.SendRecv):
        media_count += 1
    data_media_count = int(desc.has_application()) if not self.should_add_data_channel else 0
    self.expected_number_of_incoming_media = media_count + data_media_count
    if self.expected_number_of_incoming_media == 0:
      self._set_event(self.incoming_media_ready_event)

  def has_incoming_video_track(self, camera_type: str) -> bool:
    return camera_type in self.incoming_camera_tracks

  def has_incoming_audio_track(self) -> bool:
    return len(self.incoming_audio_tracks) > 0

  def has_messaging_channel(self) -> bool:
    return self.messaging_channel is not None

  def get_incoming_video_track(self, camera_type: str) -> Track:
    assert camera_type in self.incoming_camera_tracks, "Video tracks are not enabled on this stream"
    assert self.is_started, "Stream must be started"
    return self.incoming_camera_tracks[camera_type]

  def get_incoming_audio_track(self) -> Track:
    assert len(self.incoming_audio_tracks) > 0, "Audio tracks are not enabled on this stream"
    assert self.is_started, "Stream must be started"
    return self.incoming_audio_tracks[0]

  def get_messaging_channel(self) -> DataChannel:
    assert self.messaging_channel is not None, "Messaging channel is not enabled on this stream"
    assert self.is_started, "Stream must be started"
    return self.messaging_channel

  def get_receiver_report_stats(self) -> Dict[str, RtcpReceiverReport]:
    return dict(self._receiver_reports)

  def set_message_handler(self, message_handler: MessageHandler):
    self.incoming_message_handlers.append(message_handler)

  @property
  def is_started(self) -> bool:
    return self.peer_connection is not None and \
           self.peer_connection.local_description() is not None and \
           self.peer_connection.remote_description() is not None and \
           self.peer_connection.state() != PeerConnection.State.Closed

  @property
  def is_connected_and_ready(self) -> bool:
    return self.peer_connection is not None and \
           self.peer_connection.state() == PeerConnection.State.Connected and \
           (self.expected_number_of_incoming_media == 0 or self.incoming_media_ready_event.is_set())

  async def _wait_for_gathering_complete(self):
    if self.peer_connection.gathering_state() != PeerConnection.GatheringState.Complete:
      await self.gathering_complete_event.wait()

  async def _send_track_loop(self, rtc_track: Track, producer_track: TiciVideoStreamTrack, rtp_config: RtpPacketizationConfig):
    while True:
      if not rtc_track.is_open():
        await asyncio.sleep(0.01)
        continue

      try:
        packet = await producer_track.recv()
        data = bytes(packet)
        if not data:
          continue

        pts = int(packet.pts or 0)
        timestamp = (rtp_config.start_timestamp + pts) & 0xFFFFFFFF
        rtc_track.send_frame(data, FrameInfo(timestamp))
      except asyncio.CancelledError:
        raise
      except Exception:
        self.logger.exception("Error in send track loop for track %s", producer_track.id)
        self._set_event(self.connection_stopped_event)
        break

  async def _receiver_report_loop(self):
    while True:
      for camera_type, (rtc_track, ssrc) in self._receiver_report_tracks.items():
        for _ in range(32):
          try:
            message = rtc_track.receive()
            if message is None: # go until queue empty (bounded to 32)
              break
            if isinstance(message, bytes):
              self._update_receiver_report(camera_type, ssrc, message)
          except asyncio.CancelledError:
            raise
          except Exception:
            self.logger.exception("Error receiving report for %s", camera_type)
            break
      await asyncio.sleep(0.05)

  def _start_sender_tasks(self):
    for rtc_track, producer_track, rtp_config in self._track_state:
      self._sender_tasks.append(asyncio.create_task(self._send_track_loop(rtc_track, producer_track, rtp_config)))
    if self._track_state:
      self._sender_tasks.append(asyncio.create_task(self._receiver_report_loop()))

  async def wait_for_connection(self):
    assert self.is_started
    await self.connection_attempted_event.wait()
    if self.peer_connection.state() != PeerConnection.State.Connected:
      raise ValueError("Connection failed.")
    if self.expected_number_of_incoming_media:
      await self.incoming_media_ready_event.wait()
    if self.messaging_channel is not None:
      await self.messaging_channel_ready_event.wait()
    self._start_sender_tasks()

  async def wait_for_disconnection(self):
    assert self.is_connected_and_ready, "Stream is not connected/ready yet (make sure wait_for_connection was awaited)"
    await self.connection_stopped_event.wait()

  async def stop(self):
    for task in self._sender_tasks:
      task.cancel()
    for task in self._sender_tasks:
      with contextlib.suppress(asyncio.CancelledError):
        await task
    self._sender_tasks.clear()
    self.peer_connection.close()
    self.peer_connection.reset_callbacks()
    self.messaging_channel = None
    self.incoming_camera_tracks.clear()
    self.incoming_audio_tracks.clear()
    self._consumer_tracks.clear()
    self._track_state.clear()
    self._receiver_reports.clear()
    self._receiver_report_tracks.clear()

  @abc.abstractmethod
  async def start(self) -> RTCSessionDescription:
    raise NotImplementedError


class WebRTCOfferStream(WebRTCBaseStream):
  def __init__(self, session_provider: ConnectionProvider, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.session_provider = session_provider

  async def start(self) -> RTCSessionDescription:
    self._loop = asyncio.get_running_loop()
    self._add_consumer_transceivers()
    if self.should_add_data_channel:
      self._add_messaging_channel()

    self.peer_connection.set_local_description(Description.Type.Offer)
    await self._wait_for_gathering_complete()
    actual_offer = self.peer_connection.local_description()

    streaming_offer = StreamingOffer(
      sdp=str(actual_offer),
      video=list(self.expected_incoming_camera_types),
    )
    remote_answer = await self.session_provider(streaming_offer)
    self._parse_incoming_streams(remote_sdp=remote_answer.sdp)
    self.peer_connection.set_remote_description(Description(remote_answer.sdp, Description.Type.Answer))
    self._on_after_media()
    actual_answer = self.peer_connection.remote_description()

    return RTCSessionDescription(str(actual_answer), actual_answer.type_string())


class WebRTCAnswerStream(WebRTCBaseStream):
  def __init__(self, session: RTCSessionDescription, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.session = session

  async def start(self) -> RTCSessionDescription:
    self._loop = asyncio.get_running_loop()
    assert self.peer_connection.remote_description() is None, "Connection already established"

    self._parse_incoming_streams(remote_sdp=self.session.sdp)
    self.peer_connection.set_remote_description(Description(self.session.sdp, Description.Type.Offer))
    self._add_producer_tracks(self.session.sdp)

    self.peer_connection.set_local_description(Description.Type.Answer)
    await self._wait_for_gathering_complete()
    actual_answer = self.peer_connection.local_description()

    return RTCSessionDescription(str(actual_answer), actual_answer.type_string())
