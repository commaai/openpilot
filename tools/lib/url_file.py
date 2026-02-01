import logging
import os
import re
import socket
import time
from hashlib import md5
from urllib3 import PoolManager, Retry
from urllib3.response import BaseHTTPResponse
from urllib3.util import Timeout

from openpilot.common.utils import atomic_write
from openpilot.system.hardware.hw import Paths
from urllib3.exceptions import MaxRetryError

#  Cache chunk size
K = 1000
CHUNK_SIZE = 1000 * K
CACHE_SIZE = 10 * 1024 * 1024 * 1024  # total cache size in GB

logging.getLogger("urllib3").setLevel(logging.WARNING)


def hash_url(link: str) -> str:
  return md5((link.split("?")[0]).encode('utf-8')).hexdigest()


def prune_cache(new_entry: str | None = None) -> None:
  """Evicts oldest cache files (LRU) until cache is under the size limit."""
  # we use a manifest to avoid tons of os.stat syscalls (slow)
  manifest = {}
  manifest_path = Paths.download_cache_root() + "manifest.txt"
  if os.path.exists(manifest_path):
    with open(manifest_path) as f:
      manifest = {parts[0]: int(parts[1]) for line in f if (parts := line.strip().split()) and len(parts) == 2}

  if new_entry:
    manifest[new_entry] = int(time.time())  # noqa: TID251

  # evict the least recently used files until under limit
  sorted_items = sorted(manifest.items(), key=lambda x: x[1])
  while len(manifest) * CHUNK_SIZE > CACHE_SIZE and sorted_items:
    key, _ = sorted_items.pop(0)
    try:
      os.remove(Paths.download_cache_root() + key)
    except OSError:
      pass
    manifest.pop(key, None)

  # write out manifest
  with atomic_write(manifest_path, mode="w", overwrite=True) as f:
    f.write('\n'.join(f"{k} {v}" for k, v in manifest.items()))

class URLFileException(Exception):
  pass


class URLFile:
  _pool_manager: PoolManager | None = None

  @staticmethod
  def reset() -> None:
    URLFile._pool_manager = None

  @staticmethod
  def pool_manager() -> PoolManager:
    if URLFile._pool_manager is None:
      socket_options = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
      retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[409, 429, 503, 504])
      URLFile._pool_manager = PoolManager(num_pools=10, maxsize=100, socket_options=socket_options, retries=retries)
    return URLFile._pool_manager

  def __init__(self, url: str, timeout: int = 10, cache: bool | None = None):
    self._url = url
    self._timeout = Timeout(connect=timeout, read=timeout)
    self._pos = 0
    self._length: int | None = None
    #  Caching enabled by default, can be disabled with DISABLE_FILEREADER_CACHE=1, or overwritten by the cache input
    self._force_download = int(os.environ.get("DISABLE_FILEREADER_CACHE", "0")) == 1
    if cache is not None:
      self._force_download = not cache

    if not self._force_download:
      os.makedirs(Paths.download_cache_root(), exist_ok=True)

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback) -> None:
    pass

  def _request(self, method: str, url: str, headers: dict[str, str] | None = None) -> BaseHTTPResponse:
    try:
      return URLFile.pool_manager().request(method, url, timeout=self._timeout, headers=headers)
    except MaxRetryError as e:
      raise URLFileException(f"Failed to {method} {url}: {e}") from e

  def get_length_online(self) -> int:
    response = self._request('HEAD', self._url)
    if not (200 <= response.status <= 299):
      return -1
    length = response.headers.get('content-length', 0)
    return int(length)

  def get_length(self) -> int:
    if self._length is not None:
      return self._length

    file_length_path = os.path.join(Paths.download_cache_root(), hash_url(self._url) + "_length")
    if not self._force_download and os.path.exists(file_length_path):
      with open(file_length_path) as file_length:
        content = file_length.read()
        self._length = int(content)
        return self._length

    self._length = self.get_length_online()
    if not self._force_download and self._length != -1:
      with atomic_write(file_length_path, mode="w", overwrite=True) as file_length:
        file_length.write(str(self._length))
    return self._length

  def read(self, ll: int | None = None) -> bytes:
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
      file_name = hash_url(self._url) + "_" + str(chunk_number)
      full_path = os.path.join(Paths.download_cache_root(), str(file_name))
      data = None
      #  If we don't have a file, download it
      if not os.path.exists(full_path):
        data = self.read_aux(ll=CHUNK_SIZE)
        with atomic_write(full_path, mode="wb", overwrite=True) as new_cached_file:
          new_cached_file.write(data)
        prune_cache(file_name)
      else:
        with open(full_path, "rb") as cached_file:
          data = cached_file.read()

      response += data[max(0, file_begin - position): min(CHUNK_SIZE, file_end - position)]

      position += CHUNK_SIZE
      if position >= file_end:
        self._pos = file_end
        return response

  def read_aux(self, ll: int | None = None) -> bytes:
    if ll is None:
      length = self.get_length()
      if length == -1:
        raise URLFileException(f"Remote file is empty or doesn't exist: {self._url}")
      end = length
    else:
      end = self._pos + ll
    data = self.get_multi_range([(self._pos, end)])
    self._pos += len(data[0])
    return data[0]

  def get_multi_range(self, ranges: list[tuple[int, int]]) -> list[bytes]:
    # HTTP range requests are inclusive
    assert all(e > s for s, e in ranges), "Range end must be greater than start"
    rs = [f"{s}-{e-1}" for s, e in ranges if e > s]

    r = self._request("GET", self._url, headers={"Range": "bytes=" + ",".join(rs)})
    if r.status not in [200, 206]:
      raise URLFileException(f"Expected 206 or 200 response {r.status} ({self._url})")

    ctype = (r.headers.get("content-type") or "").lower()
    if "multipart/byteranges" not in ctype:
      return [r.data,]

    m = re.search(r'boundary="?([^";]+)"?', ctype)
    if not m:
      raise URLFileException(f"Missing multipart boundary ({self._url})")
    boundary = m.group(1).encode()

    parts = []
    for chunk in r.data.split(b"--" + boundary):
      if b"\r\n\r\n" not in chunk:
        continue
      payload = chunk.split(b"\r\n\r\n", 1)[1].rstrip(b"\r\n")
      if payload and payload != b"--":
        parts.append(payload)
    if len(parts) != len(ranges):
      raise URLFileException(f"Expected {len(ranges)} parts, got {len(parts)} ({self._url})")
    return parts

  def seekable(self) -> bool:
    return True

  def seek(self, pos: int, whence: int = 0) -> int:
    pos = int(pos)
    if whence == os.SEEK_SET:
      self._pos = pos
    elif whence == os.SEEK_CUR:
      self._pos += pos
    elif whence == os.SEEK_END:
      length = self.get_length()
      assert length != -1, "Cannot seek from end on unknown length file"
      self._pos = length + pos
    else:
      raise URLFileException("Invalid whence value")
    return self._pos

  def tell(self) -> int:
    return self._pos

  @property
  def name(self) -> str:
    return self._url


os.register_at_fork(after_in_child=URLFile.reset)
