#!/usr/bin/env python3

import pytest

import aiortc
from aiortc.mediastreams import AudioStreamTrack

from teleoprtc.builder import WebRTCOfferBuilder, WebRTCAnswerBuilder
from teleoprtc.info import parse_info_from_offer
from teleoprtc.tracks import TiciVideoStreamTrack


class OfferCapture:
  def __init__(self):
    self.offer = None

  async def __call__(self, offer):
    self.offer = offer
    raise Exception("Offer captured")


class DummyH264VideoStreamTrack(TiciVideoStreamTrack):
  kind = "video"

  async def recv(self):
    raise NotImplementedError()

  def codec_preference(self):
    return "H264"


@pytest.mark.asyncio
class TestOfferStream:
  async def test_offer_stream_sdp_recvonly_audio(self):
    capture = OfferCapture()
    builder = WebRTCOfferBuilder(capture)
    builder.offer_to_receive_audio_stream()
    stream = builder.stream()

    try:
      _ = await stream.start()
    except Exception:
      pass

    info = parse_info_from_offer(capture.offer.sdp)
    assert info.expected_audio_track
    assert not info.incoming_audio_track

  async def test_offer_stream_sdp_sendonly_audio(self):
    capture = OfferCapture()
    builder = WebRTCOfferBuilder(capture)
    builder.add_audio_stream(AudioStreamTrack())
    stream = builder.stream()

    try:
      _ = await stream.start()
    except Exception:
      pass

    info = parse_info_from_offer(capture.offer.sdp)
    assert not info.expected_audio_track
    assert info.incoming_audio_track

  async def test_offer_stream_sdp_channel(self):
    capture = OfferCapture()
    builder = WebRTCOfferBuilder(capture)
    builder.add_messaging()
    stream = builder.stream()

    try:
      _ = await stream.start()
    except Exception:
      pass

    info = parse_info_from_offer(capture.offer.sdp)
    assert info.incoming_datachannel


@pytest.mark.asyncio
class TestAnswerStream:
  async def test_codec_preference(self):
    offer_sdp = """v=0
o=- 3910274679 3910274679 IN IP4 0.0.0.0
s=-
t=0 0
a=group:BUNDLE 0
a=msid-semantic:WMS *
m=video 1337 UDP/TLS/RTP/SAVPF 97 98 99 100 101 102
c=IN IP4 0.0.0.0
a=recvonly
a=mid:0
a=msid:34803878-98f8-4245-b45c-f773e5f926df 881dbc20-356a-499c-b4e8-695303bb901d
a=rtcp:9 IN IP4 0.0.0.0
a=rtcp-mux
a=ssrc-group:FID 1303546896 3784011659
a=ssrc:1303546896 cname:a59185ac-c115-48d3-b39b-db7d615a6966
a=ssrc:3784011659 cname:a59185ac-c115-48d3-b39b-db7d615a6966
a=rtpmap:97 VP8/90000
a=rtcp-fb:97 nack
a=rtcp-fb:97 nack pli
a=rtcp-fb:97 goog-remb
a=rtpmap:99 H264/90000
a=rtcp-fb:99 nack
a=rtcp-fb:99 nack pli
a=rtcp-fb:99 goog-remb
a=fmtp:99 level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42001f
a=ice-ufrag:1234
a=ice-pwd:1234
a=fingerprint:sha-256 15:F3:F0:23:67:44:EE:2C:AA:8C:D9:50:95:26:42:7C:67:EA:1F:D2:92:C5:97:01:7B:2E:57:C9:A3:13:00:4A
a=setup:actpass"""

    builder = WebRTCAnswerBuilder(offer_sdp)
    builder.add_video_stream("road", DummyH264VideoStreamTrack("road", 0.05))
    stream = builder.stream()
    answer = await stream.start()

    sdp_desc = aiortc.sdp.SessionDescription.parse(answer.sdp)
    video_desc = [m for m in sdp_desc.media if m.kind == "video"][0]
    codecs = video_desc.rtp.codecs
    assert codecs[0].mimeType == "video/H264"

  async def test_fail_if_preferred_codec_not_in_offer(self):
    offer_sdp = """v=0
o=- 3910274679 3910274679 IN IP4 0.0.0.0
s=-
t=0 0
a=group:BUNDLE 0
a=msid-semantic:WMS *
m=video 1337 UDP/TLS/RTP/SAVPF 97 98 99 100 101 102
c=IN IP4 0.0.0.0
a=recvonly
a=mid:0
a=msid:34803878-98f8-4245-b45c-f773e5f926df 881dbc20-356a-499c-b4e8-695303bb901d
a=rtcp:9 IN IP4 0.0.0.0
a=rtcp-mux
a=ssrc-group:FID 1303546896 3784011659
a=ssrc:1303546896 cname:a59185ac-c115-48d3-b39b-db7d615a6966
a=ssrc:3784011659 cname:a59185ac-c115-48d3-b39b-db7d615a6966
a=rtpmap:97 VP8/90000
a=rtcp-fb:97 nack
a=rtcp-fb:97 nack pli
a=rtcp-fb:97 goog-remb
a=ice-ufrag:1234
a=ice-pwd:1234
a=fingerprint:sha-256 15:F3:F0:23:67:44:EE:2C:AA:8C:D9:50:95:26:42:7C:67:EA:1F:D2:92:C5:97:01:7B:2E:57:C9:A3:13:00:4A
a=setup:actpass"""

    builder = WebRTCAnswerBuilder(offer_sdp)
    builder.add_video_stream("road", DummyH264VideoStreamTrack("road", 0.05))
    stream = builder.stream()

    with pytest.raises(ValueError):
      _ = await stream.start()
