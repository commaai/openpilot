#!/usr/bin/env python3
from functools import wraps
import http.server
import os
import threading
import time
import unittest

from parameterized import parameterized

from openpilot.tools.lib.url_file import URLFile


class CachingTestRequestHandler(http.server.BaseHTTPRequestHandler):
  FILE_EXISTS = True

  def do_GET(self):
    if self.FILE_EXISTS:
      self.send_response(200, b'1234')
    else:
      self.send_response(404)
    self.end_headers()

  def do_HEAD(self):
    if self.FILE_EXISTS:
      self.send_response(200)
      self.send_header("Content-Length", "4")
    else:
      self.send_response(404)
    self.end_headers()


class CachingTestServer(threading.Thread):
  def run(self):
    self.server = http.server.HTTPServer(("127.0.0.1", 0), CachingTestRequestHandler)
    self.port = self.server.server_port
    self.server.serve_forever()

  def stop(self):
    self.server.server_close()
    self.server.shutdown()

def with_caching_server(func):
  @wraps(func)
  def wrapper(*args, **kwargs):
    server = CachingTestServer()
    server.start()
    time.sleep(0.25) # wait for server to get it's port
    try:
      func(*args, **kwargs, port=server.port)
    finally:
      server.stop()
  return wrapper


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
  @with_caching_server
  def test_recover_from_missing_file(self, cache_enabled, port):
    os.environ["FILEREADER_CACHE"] = "1" if cache_enabled else "0"

    file_url = f"http://localhost:{port}/test.png"

    CachingTestRequestHandler.FILE_EXISTS = False
    length = URLFile(file_url).get_length()
    self.assertEqual(length, -1)

    CachingTestRequestHandler.FILE_EXISTS = True
    length = URLFile(file_url).get_length()
    self.assertEqual(length, 4)



if __name__ == "__main__":
  unittest.main()
