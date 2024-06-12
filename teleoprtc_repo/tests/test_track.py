#!/usr/bin/env python3

import unittest

import aiortc

from teleoprtc.tracks import video_track_id, parse_video_track_id, TiciVideoStreamTrack, TiciTrackWrapper


class TestTracks(unittest.TestCase):
  def test_track_id(self):
    expected_camera_type, expected_track_id = "driver", "test"
    track_id = video_track_id(expected_camera_type, expected_track_id)
    camera_type, track_id = parse_video_track_id(track_id)
    self.assertEqual(expected_camera_type, camera_type)
    self.assertEqual(expected_track_id, track_id)

  def test_track_id_invalid(self):
    with self.assertRaises(ValueError):
      parse_video_track_id("test")

  def test_tici_track_id(self):
    class VideoStream(TiciVideoStreamTrack):
      async def recv(self):
        raise NotImplementedError()

    track = VideoStream("driver", 0.1)
    camera_type, _ = parse_video_track_id(track.id)
    self.assertEqual("driver", camera_type)

  def test_tici_wrapper_id(self):
    track = TiciTrackWrapper("driver", aiortc.mediastreams.VideoStreamTrack())
    camera_type, _ = parse_video_track_id(track.id)
    self.assertEqual("driver", camera_type)


if __name__ == '__main__':
  unittest.main()
