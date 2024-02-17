import logging
import os
import socket
import time
from hashlib import sha256
from urllib3 import PoolManager, Retry
from urllib3.response import BaseHTTPResponse
from urllib3.util import Timeout

from openpilot.common.file_helpers import atomic_write_in_dir
from openpilot.system.hardware.hw import Paths
#  Cache chunk size
K = 1000
CHUNK_SIZE = 1000 * K

logging.getLogger("urllib3").setLevel(logging.WARNING)

def hash_256(link: str) -> str:
  hsh = str(sha256((link.split("?")[0]).encode('utf-8')).hexdigest())
  return hsh


class URLFileException(Exception):
  pass


class URLFile:
  _pool_manager: PoolManager|None = None

  @staticmethod
  def reset() -> None:
    URLFile._pool_manager = None

  @staticmethod
  def pool_manager() -> PoolManager:
    if URLFile._pool_manager is None:
      socket_options = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),]
      retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[409, 429, 503, 504])
      URLFile._pool_manager = PoolManager(num_pools=10, maxsize=100, socket_options=socket_options, retries=retries)
    return URLFile._pool_manager

  def __init__(self, url: str, timeout: int=10, debug: bool=False, cache: bool|None=None):
    self._url = url
    self._timeout = Timeout(connect=timeout, read=timeout)
    self._pos = 0
    self._length: int|None = None
    self._debug = debug
    #  True by default, false if FILEREADER_CACHE is defined, but can be overwritten by the cache input
    self._force_download = not int(os.environ.get("FILEREADER_CACHE", "0"))
    if cache is not None:
      self._force_download = not cache

    if not self._force_download:
      os.makedirs(Paths.download_cache_root(), exist_ok=True)

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback) -> None:
    pass

  def _request(self, method: str, url: str, headers: dict[str, str]|None=None) -> BaseHTTPResponse:
    return URLFile.pool_manager().request(method, url, timeout=self._timeout, headers=headers)

  def get_length_online(self) -> int:
    response = self._request('HEAD', self._url)
    if not (200 <= response.status <= 299):
      return -1
    length = response.headers.get('content-length', 0)
    return int(length)

  def get_length(self) -> int:
    if self._length is not None:
      return self._length

    file_length_path = os.path.join(Paths.download_cache_root(), hash_256(self._url) + "_length")
    if not self._force_download and os.path.exists(file_length_path):
      with open(file_length_path) as file_length:
        content = file_length.read()
        self._length = int(content)
        return self._length

    self._length = self.get_length_online()
    if not self._force_download and self._length != -1:
      with atomic_write_in_dir(file_length_path, mode="w") as file_length:
        file_length.write(str(self._length))
    return self._length

  def read(self, ll: int|None=None) -> bytes:
    if self._force_download:
      return self.read_aux(ll=ll)

    file_begin = self._pos
    file_end = self._pos + ll if ll is not None else self.get_length()
    assert file_end != -1, f"Remote file is empty or doesn't exist: {self._url}"
    #  We have to align with chunks we store. Position is the begginiing of the latest chunk that starts before or at our file
    position = (file_begin // CHUNK_SIZE) * CHUNK_SIZE
    response = b""
    while True:
      self._pos = position
      chunk_number = self._pos / CHUNK_SIZE
      file_name = hash_256(self._url) + "_" + str(chunk_number)
      full_path = os.path.join(Paths.download_cache_root(), str(file_name))
      data = None
      #  If we don't have a file, download it
      if not os.path.exists(full_path):
        data = self.read_aux(ll=CHUNK_SIZE)
        with atomic_write_in_dir(full_path, mode="wb") as new_cached_file:
          new_cached_file.write(data)
      else:
        with open(full_path, "rb") as cached_file:
          data = cached_file.read()

      response += data[max(0, file_begin - position): min(CHUNK_SIZE, file_end - position)]

      position += CHUNK_SIZE
      if position >= file_end:
        self._pos = file_end
        return response

  def read_aux(self, ll: int|None=None) -> bytes:
    download_range = False
    headers = {}
    if self._pos != 0 or ll is not None:
      if ll is None:
        end = self.get_length() - 1
      else:
        end = min(self._pos + ll, self.get_length()) - 1
      if self._pos >= end:
        return b""
      headers['Range'] = f"bytes={self._pos}-{end}"
      download_range = True

    if self._debug:
      t1 = time.time()

    response = self._request('GET', self._url, headers=headers)
    ret = response.data

    if self._debug:
      t2 = time.time()
      if t2 - t1 > 0.1:
        print(f"get {self._url} {headers!r} {t2 - t1:.3f} slow")

    response_code = response.status
    if response_code == 416:  # Requested Range Not Satisfiable
      raise URLFileException(f"Error, range out of bounds {response_code} {headers} ({self._url}): {repr(ret)[:500]}")
    if download_range and response_code != 206:  # Partial Content
      raise URLFileException(f"Error, requested range but got unexpected response {response_code} {headers} ({self._url}): {repr(ret)[:500]}")
    if (not download_range) and response_code != 200:  # OK
      raise URLFileException(f"Error {response_code} {headers} ({self._url}): {repr(ret)[:500]}")

    self._pos += len(ret)
    return ret

  def seek(self, pos:int) -> None:
    self._pos = pos

  @property
  def name(self) -> str:
    return self._url


os.register_at_fork(after_in_child=URLFile.reset)
