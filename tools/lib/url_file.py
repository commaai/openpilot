import logging
import os
import time
import threading
from hashlib import sha256
from urllib3 import PoolManager
from urllib3.util import Timeout
from tenacity import retry, wait_random_exponential, stop_after_attempt

from openpilot.common.file_helpers import atomic_write_in_dir
from openpilot.system.hardware.hw import Paths
#  Cache chunk size
K = 1000
CHUNK_SIZE = 1000 * K

logging.getLogger("urllib3").setLevel(logging.WARNING)

def hash_256(link):
  hsh = str(sha256((link.split("?")[0]).encode('utf-8')).hexdigest())
  return hsh


class URLFileException(Exception):
  pass


class URLFile:
  _tlocal = threading.local()

  def __init__(self, url, debug=False, cache=None):
    self._url = url
    self._pos = 0
    self._length = None
    self._local_file = None
    self._debug = debug
    #  True by default, false if FILEREADER_CACHE is defined, but can be overwritten by the cache input
    self._force_download = not int(os.environ.get("FILEREADER_CACHE", "0"))
    if cache is not None:
      self._force_download = not cache

    if not self._force_download:
      os.makedirs(Paths.download_cache_root(), exist_ok=True)

    try:
      self._http_client = URLFile._tlocal.http_client
    except AttributeError:
      self._http_client = URLFile._tlocal.http_client = PoolManager()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    if self._local_file is not None:
      os.remove(self._local_file.name)
      self._local_file.close()
      self._local_file = None

  @retry(wait=wait_random_exponential(multiplier=1, max=5), stop=stop_after_attempt(3), reraise=True)
  def get_length_online(self):
    timeout = Timeout(connect=50.0, read=500.0)
    response = self._http_client.request('HEAD', self._url, timeout=timeout, preload_content=False)
    if not (200 <= response.status <= 299):
      return -1
    length = response.headers.get('content-length', 0)
    return int(length)

  def get_length(self):
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

  def read(self, ll=None):
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

  @retry(wait=wait_random_exponential(multiplier=1, max=5), stop=stop_after_attempt(3), reraise=True)
  def read_aux(self, ll=None):
    download_range = False
    headers = {'Connection': 'keep-alive'}
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

    timeout = Timeout(connect=50.0, read=500.0)
    response = self._http_client.request('GET', self._url, timeout=timeout, preload_content=False, headers=headers)
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

  def seek(self, pos):
    self._pos = pos

  @property
  def name(self):
    return self._url
