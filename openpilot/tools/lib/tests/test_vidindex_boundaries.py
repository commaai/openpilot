from unittest.mock import patch, mock_open

import pytest

from openpilot.tools.lib.vidindex import VideoFileInvalid, hevc_index


def index(data):
  with patch("openpilot.tools.lib.vidindex.FileReader", mock_open(read_data=data)):
    return hevc_index("camera.hevc")


def test_three_byte_start_code():
  data = bytes.fromhex("00 00 01 02 01 e0")
  assert index(data)[0] == [(0, 0)]


def test_leading_zero_bytes():
  data = bytes.fromhex("00 00 00 00 00 01 02 01 e0")
  assert index(data)[0] == [(0, 3)]


@pytest.mark.parametrize("header", ["82 01", "02 00"])
def test_invalid_nal_header(header):
  with pytest.raises(VideoFileInvalid):
    index(bytes.fromhex("00 00 00 01 " + header + " e0"))


def test_empty_slice_reports_video_error():
  with pytest.raises(VideoFileInvalid):
    index(bytes.fromhex("00 00 00 01 02 01"))


def test_slice_header_cannot_consume_next_unit():
  with pytest.raises(VideoFileInvalid):
    index(bytes.fromhex("00 00 00 01 02 01 00 00 01 40 01 ff"))
