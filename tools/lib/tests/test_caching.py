#!/usr/bin/env python3

import os
import unittest
from tools.lib.url_file import URLFile


class TestFileDownload(unittest.TestCase):

    def compare_loads(self, url, start=0, length=None):
        file_cached = URLFile(url, cache=True)
        file_downloaded = URLFile(url, cache=False)
        file_cached.seek(start)
        file_downloaded.seek(start)
        self.assertEqual(file_cached.get_length(), file_downloaded.get_length())
        self.assertLess(length if length is not None else 0, file_downloaded.get_length())
        if length is None:
            response_cached = file_cached.read()
            response_downloaded = file_downloaded.read()
        else:
            response_cached = file_cached.read(ll=length)
            response_downloaded = file_downloaded.read(ll=length)

        self.assertEqual(response_cached, response_downloaded)

    def test_downloads(self):
        # Make sure we don't force cache
        os.environ["FILEREADER_CACHE"] = ""
        small_file_url = "https://raw.githubusercontent.com/commaai/openpilot/master/SAFETY.md"
        large_file_url = "https://commadataci.blob.core.windows.net/openpilotci/0375fdf7b1ce594d/2019-06-13--08-32-25/3/qlog.bz2"
        #  If you want large file to be larger than a chunk
        #  large_file_url = "https://commadataci.blob.core.windows.net/openpilotci/0375fdf7b1ce594d/2019-06-13--08-32-25/3/fcamera.hevc"

        #  Load full small file
        self.compare_loads(small_file_url)

        #  Load the end 100 bytes of both files
        file_small = URLFile(small_file_url)
        length = file_small.get_length()
        self.compare_loads(small_file_url, length-100, 100)

        #  Load the bytes from 50 to 150 of both files
        self.compare_loads(small_file_url, 50, 100)
        #  Load small file 100 bytes at a time
        for i in range(length // 100):
            self.compare_loads(small_file_url, 100*i, 100)

        #  Load the end 100 bytes of both files
        file_large = URLFile(small_file_url)
        length = file_large.get_length()
        self.compare_loads(large_file_url, length-100, 100)

        #  Load full large file
        self.compare_loads(large_file_url)


if __name__ == "__main__": 
    unittest.main()
