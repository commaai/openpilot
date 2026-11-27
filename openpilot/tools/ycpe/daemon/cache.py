"""The download cache shared with openpilot's tools (Paths.download_cache_root()), in url_file.py's format: md5(url)_<n>.0
in 1 MB chunks plus md5(url)_length, keyed by the URL without its query, so signed URLs of one file share a key, and
held under its size by url_file's LRU. LogReader, FrameReader and process_replay read what the page downloaded, and
the reverse. A file is looked up under the canonical URL of every source that could hold it (internal, comma API blob,
openpilotci).
"""
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from openpilot.common.hardware.hw import Paths
from openpilot.common.utils import atomic_write
from openpilot.tools.lib import filereader
from openpilot.tools.lib.azure_container import AzureContainer
from openpilot.tools.lib.openpilotci import get_url as openpilotci_url
from openpilot.tools.lib.openpilotcontainers import DataProdContainer
from openpilot.tools.lib.url_file import CHUNK_SIZE, URLFile, hash_url, prune_cache

QLOGS = AzureContainer("commadata2", "qlog")  # where the API keeps qlogs and qcameras; the rest is in DataProdContainer
# internal, api and openpilotci hold the same bytes for a file, so a copy from any of them stands in for the others
# (and local directories hold them too)
SHARED = ("internal", "api", "openpilotci")


@dataclass(frozen=True)
class Key:
  dongle: str
  log_id: str
  seg: int
  file: str

  @property
  def route(self) -> str:
    return f"{self.dongle}|{self.log_id}"

  def __str__(self) -> str:
    return f"{self.dongle}/{self.log_id}/{self.seg}/{self.file}"


def root() -> Path:
  return Path(Paths.download_cache_root())


def canonical_url(source: str, key: Key) -> str | None:
  """The query-less URL the shared cache knows this file by when it came from source."""
  if source == "internal":
    return f"{filereader.DATA_ENDPOINT.rstrip('/')}/{key}"
  if source == "api":
    return (QLOGS if key.file.startswith(("qlog", "qcamera")) else DataProdContainer).get_url(key.route, key.seg, key.file)
  if source == "openpilotci":
    return openpilotci_url(key.route, key.seg, key.file)
  return None


def chunk_path(url: str, i: int) -> Path:
  return root() / f"{hash_url(url)}_{float(i)}"


def lookup(key: Key, sources=SHARED) -> tuple[str, str, int] | None:
  """(source, canonical url, length) of a complete cached copy of key from one of sources, or None."""
  for source in sources:
    if (url := canonical_url(source, key)) is None:
      continue
    try:
      length = int((root() / f"{hash_url(url)}_length").read_text())
    except (OSError, ValueError):
      continue
    if length > 0 and all(chunk_path(url, i).is_file() for i in range(-(-length // CHUNK_SIZE))):
      return source, url, length
  return None


# downloads: one per key at a time, readable while they run

class Fill:
  """Downloads url into the cache as url_file's chunks of cache_url, each one as soon as it's complete."""
  def __init__(self, key: Key, source: str, url: str, cache_url: str):
    self.key, self.source, self.url, self.cache_url = key, source, url, cache_url
    self.total: int | None = None
    self.error: str | None = None
    self.finished = threading.Event()

  def running(self) -> bool:
    if self.error:
      raise OSError(self.error)
    return not self.finished.is_set()

  def run(self) -> None:
    try:
      root().mkdir(parents=True, exist_ok=True)
      r = URLFile.pool_manager().request("GET", self.url, preload_content=False, timeout=30.0)
      try:
        if r.status != 200:
          raise OSError(f"HTTP {r.status} from {self.source}")
        if not (total := int(r.headers.get("content-length") or 0)):
          raise OSError(f"no length from {self.source}")
        self.total = total
        buf, n = bytearray(), 0
        for data in r.stream(CHUNK_SIZE):
          buf += data
          while len(buf) >= CHUNK_SIZE:
            self.write(n, buf[:CHUNK_SIZE])
            del buf[:CHUNK_SIZE]
            n += 1
      finally:
        r.release_conn()
      if (got := n * CHUNK_SIZE + len(buf)) != total:
        raise OSError(f"short read {got}/{total}")
      if buf:
        self.write(n, buf)
      with atomic_write(str(root() / f"{hash_url(self.cache_url)}_length"), overwrite=True) as out:
        out.write(str(total))
    except Exception as e:
      self.error = f"{type(e).__name__}: {e}"
    finally:
      with _fills_lock:
        _fills.pop(self.key, None)
      self.finished.set()

  def write(self, n: int, data: bytes) -> None:
    p = chunk_path(self.cache_url, n)
    with atomic_write(str(p), "wb", overwrite=True) as out:
      out.write(data)
    with _prune_lock:  # url_file's LRU over the whole cache, by its manifest
      prune_cache(p.name)


_fills: dict[Key, Fill] = {}
_fills_lock = threading.Lock()
_prune_lock = threading.Lock()


def read(url: str, start: int, end: int, fill: Fill | None = None):
  """Bytes [start, end) of url from its cached chunks, waiting on fill for the ones it hasn't written yet."""
  for i in range(start // CHUNK_SIZE, -(-end // CHUNK_SIZE)):
    p = chunk_path(url, i)
    while fill is not None and not p.is_file() and fill.running():
      time.sleep(0.01)
    base = i * CHUNK_SIZE
    yield p.read_bytes()[max(0, start - base):end - base]


def fill(key: Key, source: str, url: str) -> Fill:
  """The download of key in flight, starting one in a thread if there's none."""
  with _fills_lock:
    if (f := _fills.get(key)) is not None:
      return f
    f = _fills[key] = Fill(key, source, url, canonical_url(source, key) or url)
  threading.Thread(target=f.run, daemon=True, name=f"fill {key}").start()
  return f


def filling(key: Key) -> Fill | None:
  """The download of key in flight, from whichever source."""
  return _fills.get(key)
