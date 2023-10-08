#!/usr/bin/env python3
import os
import unittest

from pathlib import Path
from parameterized import parameterized
from unittest import mock

from openpilot.system.hardware.hw import Paths
from openpilot.tools.lib.url_file import URLFile


class TestFileDownload(unittest.TestCase):

  def compare_loads(self, url, start=0, length=None):
    """Compares range between cached and non cached version"""
    file_cached = URLFile(url, cache=True)
    file_downloaded = URLFile(url, cache=False)

    file_cached.seek(start)
    file_downloaded.seek(start)

    self.assertEqual(file_cached.get_length(), file_downloaded.get_length())
    self.assertLessEqual(length + start if length is not None else 0, file_downloaded.get_length())

    response_cached = file_cached.read(ll=length)
    response_downloaded = file_downloaded.read(ll=length)

    self.assertEqual(response_cached, response_downloaded)

    # Now test with cache in place
    file_cached = URLFile(url, cache=True)
    file_cached.seek(start)
    response_cached = file_cached.read(ll=length)

    self.assertEqual(file_cached.get_length(), file_downloaded.get_length())
    self.assertEqual(response_cached, response_downloaded)

  def test_small_file(self):
    # Make sure we don't force cache
    os.environ["FILEREADER_CACHE"] = "0"
    small_file_url = "https://raw.githubusercontent.com/commaai/openpilot/master/docs/SAFETY.md"
    #  If you want large file to be larger than a chunk
    #  large_file_url = "https://commadataci.blob.core.windows.net/openpilotci/0375fdf7b1ce594d/2019-06-13--08-32-25/3/fcamera.hevc"

    #  Load full small file
    self.compare_loads(small_file_url)

    file_small = URLFile(small_file_url)
    length = file_small.get_length()

    self.compare_loads(small_file_url, length - 100, 100)
    self.compare_loads(small_file_url, 50, 100)

    #  Load small file 100 bytes at a time
    for i in range(length // 100):
      self.compare_loads(small_file_url, 100 * i, 100)

  def test_large_file(self):
    large_file_url = "https://commadataci.blob.core.windows.net/openpilotci/0375fdf7b1ce594d/2019-06-13--08-32-25/3/qlog.bz2"
    #  Load the end 100 bytes of both files
    file_large = URLFile(large_file_url)
    length = file_large.get_length()

    self.compare_loads(large_file_url, length - 100, 100)
    self.compare_loads(large_file_url)

  @parameterized.expand([(True, ), (False, )])
  def test_recover_from_missing_file(self, cache_enabled):
    os.environ["FILEREADER_CACHE"] = "1" if cache_enabled else "0"

    file_url = "http://localhost:5001/test.png"

    file_exists = False

    def get_length_online_mock(self):
      if file_exists:
        return 4
      return -1

    patch_length = mock.patch.object(URLFile, "get_length_online", get_length_online_mock)
    patch_length.start()
    try:
      length = URLFile(file_url).get_length()
      self.assertEqual(length, -1)

      file_exists = True
      length = URLFile(file_url).get_length()
      self.assertEqual(length, 4)
    finally:
      tempfile_length = Path(Paths.download_cache_root()) / "ba2119904385654cb0105a2da174875f8e7648db175f202ecae6d6428b0e838f_length"
      if tempfile_length.exists():
        tempfile_length.unlink()
      patch_length.stop()


if __name__ == "__main__":
  unittest.main()
