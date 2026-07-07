import os
import time
from functools import cache
from hashlib import md5
from urllib.parse import urlparse

from fsspec.caching import BaseCache, register_cache
from fsspec.core import url_to_fs

from openpilot.common.hardware.hw import Paths
from openpilot.common.utils import atomic_write

K = 1000
CHUNK_SIZE = 1000 * K
CACHE_SIZE = 10 * 1024 * 1024 * 1024
CACHED_PROTOCOLS = {"http", "https", "mkv"}


class ChunkCache(BaseCache):
  name = "chunk"

  @staticmethod
  def cache_file(name: str) -> str:
    return os.path.join(Paths.download_cache_root(), name)

  @staticmethod
  def hash_url(link: str) -> str:
    return md5((link.split("?")[0]).encode('utf-8')).hexdigest()

  @staticmethod
  def prune_cache(new_entry: str | None = None) -> None:
    """Evicts oldest cache files until cache is under the size limit."""
    os.makedirs(Paths.download_cache_root(), exist_ok=True)
    manifest = {}
    manifest_path = ChunkCache.cache_file("manifest.txt")
    if os.path.exists(manifest_path):
      with open(manifest_path) as f:
        manifest = {parts[0]: int(parts[1]) for line in f if (parts := line.strip().split()) and len(parts) == 2}

    if new_entry:
      manifest[new_entry] = int(time.time())  # noqa: TID251

    sorted_items = sorted(manifest.items(), key=lambda x: x[1])
    while len(manifest) * CHUNK_SIZE > CACHE_SIZE and sorted_items:
      key, _ = sorted_items.pop(0)
      try:
        os.remove(ChunkCache.cache_file(key))
      except OSError:
        pass
      manifest.pop(key, None)

    with atomic_write(manifest_path, mode="w", overwrite=True) as f:
      f.write('\n'.join(f"{k} {v}" for k, v in manifest.items()))

  def __init__(self, blocksize: int, fetcher, size: int, cache_key: str):
    super().__init__(blocksize, fetcher, size)
    self.cache_key = cache_key
    os.makedirs(Paths.download_cache_root(), exist_ok=True)

  def _chunk_name(self, block_number: int) -> str:
    return f"{self.cache_key}_{float(block_number)}"

  def _fetch(self, start: int | None, end: int | None) -> bytes:
    start = 0 if start is None else start
    end = self.size if end is None else end
    end = min(end, self.size)
    if start >= self.size or start >= end:
      return b""

    response = []
    for block_number in range(start // self.blocksize, (end - 1) // self.blocksize + 1):
      chunk_start = block_number * self.blocksize
      chunk_end = min(chunk_start + self.blocksize, self.size)
      file_name = self._chunk_name(block_number)
      full_path = ChunkCache.cache_file(file_name)

      if os.path.exists(full_path):
        self.hit_count += 1
        with open(full_path, "rb") as cached_file:
          data = cached_file.read()
      else:
        self.miss_count += 1
        self.total_requested_bytes += chunk_end - chunk_start
        data = self.fetcher(chunk_start, chunk_end)
        with atomic_write(full_path, mode="wb", overwrite=True) as new_cached_file:
          new_cached_file.write(data)
        ChunkCache.prune_cache(file_name)

      response.append(data[max(0, start - chunk_start): min(len(data), end - chunk_start)])

    return b"".join(response)


register_cache(ChunkCache, clobber=True)


class FsspecFile:
  @staticmethod
  def _should_cache(fn: str, cache: bool | None = None) -> bool:
    return cache is not False and os.environ.get("DISABLE_FILEREADER_CACHE", "0") != "1" and urlparse(fn).scheme in CACHED_PROTOCOLS

  @staticmethod
  def _url_to_fs(fn: str):
    if urlparse(fn).scheme in {"http", "https"}:
      return url_to_fs(fn, headers={"Accept-Encoding": "identity"})
    return url_to_fs(fn)

  @staticmethod
  def _get_length(fn: str, fs, path, cache: bool | None = None) -> int:
    cache_enabled = FsspecFile._should_cache(fn, cache)
    file_length_path = ChunkCache.cache_file(ChunkCache.hash_url(fn) + "_length")

    if cache_enabled and os.path.exists(file_length_path):
      with open(file_length_path) as file_length:
        return int(file_length.read())

    try:
      length = int(fs.info(path).get("size") or 0)
    except FileNotFoundError:
      return -1

    if cache_enabled:
      os.makedirs(Paths.download_cache_root(), exist_ok=True)
      with atomic_write(file_length_path, mode="w", overwrite=True) as file_length:
        file_length.write(str(length))
    return length

  @staticmethod
  @cache
  def exists(fn: str) -> bool:
    try:
      fs, path = FsspecFile._url_to_fs(fn)
      return fs.exists(path)
    except (FileNotFoundError, OSError, ValueError):
      return False

  @property
  def file(self):
    if self._file is None:
      raise FileNotFoundError(self.name)
    return self._file

  def __init__(self, fn: str, cache: bool | None = None):
    self.name = fn
    self.fs, self.path = FsspecFile._url_to_fs(fn)
    self._length: int | None = None
    self._file = None

    cache_enabled = FsspecFile._should_cache(fn, cache)
    open_kwargs = {}
    if urlparse(fn).scheme in CACHED_PROTOCOLS:
      if cache_enabled:
        os.makedirs(Paths.download_cache_root(), exist_ok=True)

      self._length = FsspecFile._get_length(fn, self.fs, self.path, cache=cache)
      if self._length == -1:
        return

      open_kwargs["block_size"] = CHUNK_SIZE
      open_kwargs["size"] = self._length

      if cache_enabled:
        open_kwargs["cache_type"] = ChunkCache.name
        open_kwargs["cache_options"] = {"cache_key": ChunkCache.hash_url(fn)}
      else:
        open_kwargs["cache_type"] = "none"

    self._file = self.fs.open(self.path, "rb", **open_kwargs)

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.close()

  def close(self):
    if self._file is not None:
      self._file.close()

  @property
  def size(self) -> int:
    if self._length is not None:
      return self._length
    return int(self.fs.info(self.path).get("size") or 0)

  def get_length(self) -> int:
    return self.size

  def read(self, ll: int | None = None) -> bytes:
    return self.file.read(-1 if ll is None else ll)

  def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
    return self.file.seek(offset, whence)

  def tell(self) -> int:
    return self.file.tell()

  def seekable(self) -> bool:
    return self.file.seekable()

  def get_multi_range(self, ranges: list[tuple[int, int]]) -> list[bytes]:
    if len(ranges) == 0:
      return []

    starts = [start for start, _ in ranges]
    ends = [end for _, end in ranges]
    return self.fs.cat_ranges([self.path] * len(ranges), starts, ends, on_error="raise")


def FileReader(fn, cache: bool | None = None):
  return FsspecFile(fn, cache=cache)
