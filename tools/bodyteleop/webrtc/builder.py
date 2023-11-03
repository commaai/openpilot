import abc

import aiortc

from tools.bodyteleop.webrtc.stream import WebRTCBaseStream, WebRTCOfferStream, WebRTCAnswerStream, ConnectionProvider


class WebRTCStreamBuilder(abc.ABC):
  def __init__(self):
    raise NotImplemented("Use static methods (offer, answer) to create a builder")

  @abc.abstractmethod
  def request_video_stream(self, camera_type: str):
    raise NotImplemented

  @abc.abstractmethod
  def request_audio_stream(self):
    raise NotImplemented

  @abc.abstractmethod
  def add_video_stream(self, camera_type: str, track: aiortc.MediaStreamTrack):
    raise NotImplemented

  @abc.abstractmethod
  def add_audio_stream(self, track: aiortc.MediaStreamTrack):
    raise NotImplemented

  @abc.abstractmethod
  def add_messaging(self):
    raise NotImplemented

  @abc.abstractmethod
  def stream(self) -> WebRTCBaseStream:
    raise NotImplemented

  @staticmethod
  def offer(connection_provider: ConnectionProvider):
    return WebRTCOfferBuilder(connection_provider)

  @staticmethod
  def answer(sdp: str):
    return WebRTCAnswerBuilder(sdp)


class WebRTCOfferBuilder(WebRTCStreamBuilder):
  """
  Builder for WebRTC Offer
  Support:
  - audio input and output
  - video input
  - can add data channe
  """

  def __init__(self, connection_provider: ConnectionProvider):
    self.connection_provider = connection_provider
    self.requested_camera_types = []
    self.requested_audio = False
    self.audio_tracks = []
    self.messaging_enabled = False

  def request_video_stream(self, camera_type: str):
    assert camera_type in ["driver", "wideRoad", "road"]
    self.requested_camera_types.append(camera_type)

  def request_audio_stream(self):
    self.requested_audio = True

  def add_video_stream(self, camera_type: str, track: aiortc.MediaStreamTrack):
    raise NotImplemented("Adding outgoing video tracks is not supported on offer streams")

  def add_audio_stream(self, track: aiortc.MediaStreamTrack):
    assert len(self.audio_tracks) == 0
    self.audio_tracks = [track]

  def add_messaging(self):
    self.messaging_enabled = True

  def stream(self) -> WebRTCBaseStream:
    return WebRTCOfferStream(
      self.connection_provider,
      consumed_camera_types=self.requested_camera_types,
      consume_audio=self.requested_audio,
      video_producer_tracks=[],
      audio_producer_tracks=self.audio_tracks,
      should_add_data_channel=self.messaging_enabled,
    )


class WebRTCAnswerBuilder(WebRTCStreamBuilder):
  """
  Builder for WebRTC Answer
  Supports:
  - audio input and output
  - video output
  """

  def __init__(self, offer_sdp: str):
    self.offer_sdp = offer_sdp
    self.video_tracks = dict()
    self.requested_audio = False
    self.audio_tracks = []

  def request_video_stream(self, camera_type: str):
    raise NotImplemented("Requesting incoming video tracks is not supported on answer streams")

  def request_audio_stream(self):
    self.requested_audio = True

  def add_video_stream(self, camera_type: str, track: aiortc.MediaStreamTrack):
    assert camera_type not in self.video_tracks
    assert camera_type in ["driver", "wideRoad", "road"]
    self.video_tracks[camera_type] = track

  def add_audio_stream(self, track: aiortc.MediaStreamTrack):
    assert len(self.audio_tracks) == 0
    self.audio_tracks = [track]

  def add_messaging(self):
    raise NotImplemented("Messaging can be requested by offer stream only")

  def stream(self) -> WebRTCBaseStream:
    description = aiortc.RTCSessionDescription(sdp=self.offer_sdp, type="offer")
    return WebRTCAnswerStream(
      description,
      consumed_camera_types=[],
      consume_audio=self.requested_audio,
      video_producer_tracks=list(self.video_tracks.values()),
      audio_producer_tracks=self.audio_tracks,
      should_add_data_channel=False,
    )
