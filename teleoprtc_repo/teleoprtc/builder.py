import abc
from typing import Dict, List

import aiortc

from teleoprtc.stream import WebRTCBaseStream, WebRTCOfferStream, WebRTCAnswerStream, ConnectionProvider
from teleoprtc.tracks import TiciVideoStreamTrack, TiciTrackWrapper


class WebRTCStreamBuilder(abc.ABC):
  @abc.abstractmethod
  def stream(self) -> WebRTCBaseStream:
    raise NotImplementedError


class WebRTCOfferBuilder(WebRTCStreamBuilder):
  def __init__(self, connection_provider: ConnectionProvider):
    self.connection_provider = connection_provider
    self.requested_camera_types: List[str] = []
    self.requested_audio = False
    self.audio_tracks: List[aiortc.MediaStreamTrack] = []
    self.messaging_enabled = False

  def offer_to_receive_video_stream(self, camera_type: str):
    assert camera_type in ["driver", "wideRoad", "road"]
    self.requested_camera_types.append(camera_type)

  def offer_to_receive_audio_stream(self):
    self.requested_audio = True

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
  def __init__(self, offer_sdp: str):
    self.offer_sdp = offer_sdp
    self.video_tracks: Dict[str, aiortc.MediaStreamTrack] = dict()
    self.requested_audio = False
    self.audio_tracks: List[aiortc.MediaStreamTrack] = []

  def offer_to_receive_audio_stream(self):
    self.requested_audio = True

  def add_video_stream(self, camera_type: str, track: aiortc.MediaStreamTrack):
    assert camera_type not in self.video_tracks
    assert camera_type in ["driver", "wideRoad", "road"]
    if not isinstance(track, TiciVideoStreamTrack):
      track = TiciTrackWrapper(camera_type, track)
    self.video_tracks[camera_type] = track

  def add_audio_stream(self, track: aiortc.MediaStreamTrack):
    assert len(self.audio_tracks) == 0
    self.audio_tracks = [track]

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
