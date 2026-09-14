import asyncio
import os
import unittest
from unittest.mock import Mock, patch

from libdatachannel import Description
from openpilot.cereal import messaging
from openpilot.system.webrtc.device.video import LiveStreamVideoStreamTrack, V4L2_BUF_FLAG_KEYFRAME
from teleoprtc.stream import WebRTCBaseStream


def frame(frame_id, timestamp, keyframe=False):
  msg = messaging.new_message("livestreamNarrowRoadEncodeData")
  msg.livestreamNarrowRoadEncodeData.idx.frameId = frame_id
  msg.livestreamNarrowRoadEncodeData.idx.timestampSof = timestamp
  msg.livestreamNarrowRoadEncodeData.idx.flags = V4L2_BUF_FLAG_KEYFRAME if keyframe else 0
  msg.livestreamNarrowRoadEncodeData.data = b"frame"
  return msg


class TestCamera120(unittest.IsolatedAsyncioTestCase):
  def setUp(self):
    self.env = patch.dict(os.environ, CAMERA_720P120="1")
    self.env.start()
    self.addCleanup(self.env.stop)
    self.params = patch("openpilot.system.webrtc.device.video.Params", return_value=Mock())
    self.params.start()
    self.addCleanup(self.params.stop)
    self.sock = patch("openpilot.system.webrtc.device.video.messaging.sub_sock", return_value=Mock())
    self.sub_sock = self.sock.start()
    self.addCleanup(self.sock.stop)

  async def test_capture_timestamps_preserve_120fps_without_wall_clock_pacing(self):
    track = LiveStreamVideoStreamTrack("road")
    messages = [frame(i, 1_000_000_000 + round(i * 1e9 / 120), i == 0) for i in range(120)]
    with patch("openpilot.system.webrtc.device.video.messaging.recv_one_or_none", side_effect=messages):
      packets = [await track.recv() for _ in messages]
    self.assertEqual(len(packets), 120)
    self.assertLessEqual(abs(packets[-1].pts - packets[0].pts - 119 * 750), 2)
    self.assertTrue(all(b.pts > a.pts for a, b in zip(packets, packets[1:], strict=False)))
    self.sub_sock.assert_called_with("livestreamNarrowRoadEncodeData", conflate=False)

  async def test_reference_gap_waits_for_new_keyframe(self):
    track = LiveStreamVideoStreamTrack("road")
    messages = [frame(1, 100, True), frame(3, 300), frame(4, 400), frame(5, 500, True)]
    with patch("openpilot.system.webrtc.device.video.messaging.recv_one_or_none", side_effect=messages):
      await track.recv()
      await asyncio.wait_for(track.recv(), 1)
    self.assertEqual(track._last_frame_id, 5)
    track.params.put.assert_any_call("LivestreamRequestKeyframe", True, block=False)

  async def test_all_client_camera_selections_use_narrow(self):
    track = LiveStreamVideoStreamTrack("wideRoad")
    track.switch_camera("driver")
    self.sub_sock.assert_called_with("livestreamNarrowRoadEncodeData", conflate=False)
    self.assertEqual(track.h264_profile_level_id, "42e02a")

  async def test_default_mode_keeps_existing_socket(self):
    with patch.dict(os.environ, CAMERA_720P120="0"):
      track = LiveStreamVideoStreamTrack("driver")
    self.sub_sock.assert_called_with("livestreamCabinEncodeData", conflate=True)
    self.assertIsNone(track.h264_profile_level_id)

  async def test_sdp_rejects_insufficient_level_and_selects_baseline_42(self):
    video = Description.Video("video", Description.Direction.RecvOnly)
    video.add_h264_codec(96)
    video.add_h264_codec(98, "profile-level-id=42e01f;packetization-mode=1")
    desc = Description("", Description.Type.Offer)
    desc.add_media(video)
    stream = Mock()
    with self.assertRaises(ValueError):
      WebRTCBaseStream._find_offer_video(stream, str(desc), set(), "42e02a")
    video.add_h264_codec(100, "profile-level-id=42e02a;packetization-mode=1")
    desc = Description("", Description.Type.Offer)
    desc.add_media(video)
    self.assertEqual(WebRTCBaseStream._find_offer_video(stream, str(desc), set(), "42e02a"), ("video", 100))


if __name__ == "__main__":
  unittest.main()
