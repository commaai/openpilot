#!/usr/bin/env python3

from teleoprtc.info import parse_info_from_offer


def lf2crlf(x):
    return x.replace("\n", "\r\n")


class TestStream:
  def test_double_video_tracks(self):
    sdp = """v=0
o=- 3910210993 3910210993 IN IP4 0.0.0.0
s=-
t=0 0
a=group:BUNDLE 0 1
a=msid-semantic:WMS *
m=video 9 UDP/TLS/RTP/SAVPF 97 98 99 100 101 102
c=IN IP4 0.0.0.0
a=recvonly
a=extmap:1 urn:ietf:params:rtp-hdrext:sdes:mid
a=extmap:2 http://www.webrtc.org/experiments/rtp-hdrext/abs-send-time
a=mid:0
a=msid:e123f852-010c-4b7b-8761-71b72fbfd013 2b75cb0e-6b34-48d6-8bf9-21b809f2e08e
a=rtcp:9 IN IP4 0.0.0.0
a=rtcp-mux
a=ssrc-group:FID 1048118556 4149054509
a=ssrc:1048118556 cname:61992fce-bab5-42a0-ab8c-7112adfb1857
a=ssrc:4149054509 cname:61992fce-bab5-42a0-ab8c-7112adfb1857
a=rtpmap:97 VP8/90000
a=rtcp-fb:97 nack
a=rtcp-fb:97 nack pli
a=rtcp-fb:97 goog-remb
a=rtpmap:98 rtx/90000
a=fmtp:98 apt=97
a=rtpmap:99 H264/90000
a=rtcp-fb:99 nack
a=rtcp-fb:99 nack pli
a=rtcp-fb:99 goog-remb
a=fmtp:99 level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42001f
a=rtpmap:100 rtx/90000
a=fmtp:100 apt=99
a=rtpmap:101 H264/90000
a=rtcp-fb:101 nack
a=rtcp-fb:101 nack pli
a=rtcp-fb:101 goog-remb
a=fmtp:101 level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42e01f
a=rtpmap:102 rtx/90000
a=fmtp:102 apt=101
a=ice-ufrag:jxQW
a=ice-pwd:KpJ0tfaY2RxnIYpTHqPSSv
a=fingerprint:sha-256 70:3A:2D:37:3C:52:96:0E:10:F6:4D:7A:EB:18:38:1B:FD:CA:A5:90:D7:6C:DA:A9:39:76:C9:2F:FB:FF:56:0C
a=setup:actpass
m=video 9 UDP/TLS/RTP/SAVPF 97 98 99 100 101 102
c=IN IP4 0.0.0.0
a=recvonly
a=extmap:1 urn:ietf:params:rtp-hdrext:sdes:mid
a=extmap:2 http://www.webrtc.org/experiments/rtp-hdrext/abs-send-time
a=mid:1
a=msid:e123f852-010c-4b7b-8761-71b72fbfd013 311db759-8d51-479c-a5b4-5c8d055c43ec
a=rtcp:9 IN IP4 0.0.0.0
a=rtcp-mux
a=ssrc-group:FID 4096183284 2713379498
a=ssrc:4096183284 cname:61992fce-bab5-42a0-ab8c-7112adfb1857
a=ssrc:2713379498 cname:61992fce-bab5-42a0-ab8c-7112adfb1857
a=rtpmap:97 VP8/90000
a=rtcp-fb:97 nack
a=rtcp-fb:97 nack pli
a=rtcp-fb:97 goog-remb
a=rtpmap:98 rtx/90000
a=fmtp:98 apt=97
a=rtpmap:99 H264/90000
a=rtcp-fb:99 nack
a=rtcp-fb:99 nack pli
a=rtcp-fb:99 goog-remb
a=fmtp:99 level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42001f
a=rtpmap:100 rtx/90000
a=fmtp:100 apt=99
a=rtpmap:101 H264/90000
a=rtcp-fb:101 nack
a=rtcp-fb:101 nack pli
a=rtcp-fb:101 goog-remb
a=fmtp:101 level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42e01f
a=rtpmap:102 rtx/90000
a=fmtp:102 apt=101
a=ice-ufrag:1234
a=ice-pwd:1234
a=fingerprint:sha-256 70:3A:2D:37:3C:52:96:0E:10:F6:4D:7A:EB:18:38:1B:FD:CA:A5:90:D7:6C:DA:A9:39:76:C9:2F:FB:FF:56:0C
a=setup:actpass"""
    info = parse_info_from_offer(lf2crlf(sdp))
    assert info.n_expected_camera_tracks == 2
    assert not info.expected_audio_track
    assert not info.incoming_audio_track
    assert not info.incoming_datachannel

  def test_recvonly_audio(self):
    sdp = """v=0
o=- 3910210904 3910210904 IN IP4 0.0.0.0
s=-
t=0 0
a=group:BUNDLE 0
a=msid-semantic:WMS *
m=audio 9 UDP/TLS/RTP/SAVPF 96 0 8
c=IN IP4 0.0.0.0
a=recvonly
a=extmap:1 urn:ietf:params:rtp-hdrext:sdes:mid
a=extmap:2 urn:ietf:params:rtp-hdrext:ssrc-audio-level
a=mid:0
a=msid:eb1d3f1a-569a-465f-b419-319477bfded6 e44eecb2-1a04-4547-97d8-481389f50d5b
a=rtcp:9 IN IP4 0.0.0.0
a=rtcp-mux
a=ssrc:1233332626 cname:ca4dede8-4994-4a6d-9ae3-923b28177ca5
a=rtpmap:96 opus/48000/2
a=rtpmap:0 PCMU/8000
a=rtpmap:8 PCMA/8000
a=ice-ufrag:1234
a=ice-pwd:1234
a=fingerprint:sha-256 40:4B:14:CF:70:B8:67:E1:B1:FF:7E:F9:22:6E:60:7D:73:B5:1E:38:4B:10:20:9C:CD:1C:47:02:52:ED:45:25
a=setup:actpass"""
    info = parse_info_from_offer(lf2crlf(sdp))
    assert info.n_expected_camera_tracks == 0
    assert info.expected_audio_track
    assert not info.incoming_audio_track
    assert not info.incoming_datachannel

  def test_incoming_datachanel(self):
    sdp = """v=0
o=- 3910211092 3910211092 IN IP4 0.0.0.0
s=-
t=0 0
a=group:BUNDLE 0
a=msid-semantic:WMS *
m=application 9 DTLS/SCTP 5000
c=IN IP4 0.0.0.0
a=mid:0
a=sctpmap:5000 webrtc-datachannel 65535
a=max-message-size:65536
a=ice-ufrag:1234
a=ice-pwd:1234
a=fingerprint:sha-256 9B:C0:F3:35:8E:05:A1:15:DB:F8:39:0E:B0:E0:0C:EB:82:E4:B9:26:18:A6:43:2D:B9:9A:23:96:0A:59:B6:58
a=setup:actpass"""
    info = parse_info_from_offer(lf2crlf(sdp))
    assert info.n_expected_camera_tracks == 0
    assert not info.expected_audio_track
    assert not info.incoming_audio_track
    assert info.incoming_datachannel
