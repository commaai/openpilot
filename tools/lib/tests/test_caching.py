import http.server
import os
import shutil
import socket
import pytest

from openpilot.selfdrive.test.helpers import http_server_context
from openpilot.system.hardware.hw import Paths
from openpilot.tools.lib.url_file import URLFile


class CachingTestRequestHandler(http.server.BaseHTTPRequestHandler):
  FILE_EXISTS = True

  def do_GET(self):
    if self.FILE_EXISTS:
      self.send_response(206 if "Range" in self.headers else 200, b'1234')
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


@pytest.fixture
def host():
  with http_server_context(handler=CachingTestRequestHandler) as (host, port):
    yield f"http://{host}:{port}"

class TestFileDownload:

  def test_pipeline_defaults(self, host):
    # TODO: parameterize the defaults so we don't rely on hard-coded values in xx

    assert URLFile.pool_manager().pools._maxsize == 10# PoolManager num_pools param
    pool_manager_defaults = {
      "maxsize": 100,
      "socket_options": [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),],
    }
    for k, v in pool_manager_defaults.items():
      assert URLFile.pool_manager().connection_pool_kw.get(k) == v

    retry_defaults = {
      "total": 5,
      "backoff_factor": 0.5,
      "status_forcelist": [409, 429, 503, 504],
    }
    for k, v in retry_defaults.items():
      assert getattr(URLFile.pool_manager().connection_pool_kw["retries"], k) == v

    # ensure caching off by default and cache dir doesn't get created
    os.environ.pop("FILEREADER_CACHE", None)
    if os.path.exists(Paths.download_cache_root()):
      shutil.rmtree(Paths.download_cache_root())
    URLFile(f"{host}/test.txt").get_length()
    URLFile(f"{host}/test.txt").read()
    assert not os.path.exists(Paths.download_cache_root())

  def compare_loads(self, url, start=0, length=None):
    """Compares range between cached and non cached version"""
    file_cached = URLFile(url, cache=True)
    file_downloaded = URLFile(url, cache=False)

    file_cached.seek(start)
    file_downloaded.seek(start)

    assert file_cached.get_length() == file_downloaded.get_length()
    assert length + start if length is not None else 0 <= file_downloaded.get_length()

    response_cached = file_cached.read(ll=length)
    response_downloaded = file_downloaded.read(ll=length)

    assert response_cached == response_downloaded

    # Now test with cache in place
    file_cached = URLFile(url, cache=True)
    file_cached.seek(start)
    response_cached = file_cached.read(ll=length)

    assert file_cached.get_length() == file_downloaded.get_length()
    assert response_cached == response_downloaded

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

  @pytest.mark.parametrize("cache_enabled", [True, False])
  def test_recover_from_missing_file(self, host, cache_enabled):
    os.environ["FILEREADER_CACHE"] = "1" if cache_enabled else "0"

    file_url = f"{host}/test.png"

    CachingTestRequestHandler.FILE_EXISTS = False
    length = URLFile(file_url).get_length()
    assert length == -1

    CachingTestRequestHandler.FILE_EXISTS = True
    length = URLFile(file_url).get_length()
    assert length == 4
