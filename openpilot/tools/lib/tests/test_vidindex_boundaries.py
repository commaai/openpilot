from unittest import TestCase
from unittest.mock import mock_open, patch

from openpilot.tools.lib.vidindex import VideoFileInvalid, hevc_index


def index(data):
  with patch("openpilot.tools.lib.vidindex.FileReader", mock_open(read_data=data)):
    return hevc_index("camera.hevc")


class TestVidindexBoundaries(TestCase):
  def test_three_byte_start_code(self):
    data = bytes.fromhex("00 00 01 02 01 e0")
    self.assertEqual(index(data)[0], [(0, 0)])

  def test_leading_zero_bytes(self):
    data = bytes.fromhex("00 00 00 00 00 01 02 01 e0")
    self.assertEqual(index(data)[0], [(0, 3)])

  def test_forbidden_header_bit(self):
    with self.assertRaises(VideoFileInvalid):
      index(bytes.fromhex("00 00 00 01 82 01 e0"))

  def test_invalid_temporal_id(self):
    with self.assertRaises(VideoFileInvalid):
      index(bytes.fromhex("00 00 00 01 02 00 e0"))

  def test_empty_slice_reports_video_error(self):
    with self.assertRaises(VideoFileInvalid):
      index(bytes.fromhex("00 00 00 01 02 01"))

  def test_slice_header_cannot_consume_next_unit(self):
    with self.assertRaises(VideoFileInvalid):
      index(bytes.fromhex("00 00 00 01 02 01 00 00 01 40 01 ff"))
