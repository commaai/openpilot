"""Where a route's files come from, in LogReader's auto order, and a URL that serves them through the shared cache.

Order: device (pulled over ssh) → local (log dirs on this machine) → internal (DATA_ENDPOINT) → comma API →
openpilotci, the same as logreader.auto_source after the local ones. A copy already in the shared cache wins over any
download. commaCarSegments answers CORS, so the page reads it directly.
"""
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from openpilot.tools.lib.api import CommaApi
from openpilot.tools.lib.auth_config import get_token
from openpilot.tools.lib import filereader
from openpilot.tools.lib.file_downloader import cache_file_path
from openpilot.tools.lib.helpers import RE
from openpilot.tools.lib.route import FileName
from openpilot.tools.lib.url_file import URLFile
from openpilot.tools.ycpe.daemon import cache
from openpilot.tools.ycpe.daemon.cache import Key
from openpilot.tools.ycpe.daemon.core import TOKEN_PARAM, File, HTTPError, Stream, config, confine, file_chunks, module, route

SOURCES = ["device", "local", "internal", "api", "openpilotci"]
KINDS = {"rlog": FileName.RLOG, "qlog": FileName.QLOG, "qcamera": FileName.QCAMERA, "fcamera": FileName.FCAMERA,
         "ecamera": FileName.ECAMERA, "dcamera": FileName.DCAMERA}
FILES = {f: k for k, fs in KINDS.items() for f in fs}
HIT, MISS = {"X-Ycpe-Cache": "hit"}, {"X-Ycpe-Cache": "miss"}  # whether bytes came off local disk, for the page's status
API_TTL = 30 * 60  # signed URLs from the route listing are reused this long

module("sources", "1", sources=SOURCES)
_pool = ThreadPoolExecutor(16, thread_name_prefix="probe")


def parse_route(text) -> tuple[str, str]:
  """(dongle, log_id) of "dongle|log_id" (or with / or _, as route.py names routes)"""
  if not (m := re.fullmatch(RE.ROUTE_NAME, str(text).strip())):
    raise HTTPError(400, "bad route")
  return m["dongle_id"], m["log_id"]


def segments(segs, dongle: str, log_id: str):
  """The segments a request names, else the ones on this machine's disk (local log dirs, device pulls): the page
  names the rest, from the route's record"""
  if segs is None:
    have = set(local_dirs(log_id)) | set(device_segs(dongle, log_id))
    return range(max(have) + 1 if have else 0)
  if not (isinstance(segs, list) and all(isinstance(n, int) and 0 <= n < 10000 for n in segs)):
    raise HTTPError(400, "bad segs")
  return segs


# per-source lookups: each gives {seg: {file: (url or path, size)}} for what it has

_api: dict[tuple[str, str], tuple[float, dict]] = {}
_api_lock = threading.Lock()


def api_files(dongle: str, log_id: str, jwt: str | None) -> dict[int, dict[str, tuple[str, int | None]]]:
  """The route's signed file URLs from the comma API, kept for API_TTL per (route, token)."""
  k = (f"{dongle}|{log_id}", jwt or "")
  with _api_lock:
    hit = _api.get(k)
  if hit and time.monotonic() - hit[0] < API_TTL:
    return hit[1]
  listing = CommaApi(jwt or get_token()).get(f"v1/route/{dongle}|{log_id}/files")
  out: dict[int, dict] = {}
  for urls in listing.values():
    for url in urls:
      *_, seg, name = urlparse(url).path.split("/")
      out.setdefault(int(seg), {})[name] = (url, None)
  with _api_lock:
    _api[k] = (time.monotonic(), out)
  return out


def local_dirs(log_id: str) -> dict[int, Path]:
  """Segment directories named <log_id>--<n> under the configured log roots (the device's realdata layout)."""
  out = {}
  for r in config.log_roots:
    try:
      for p in Path(r).glob(f"{log_id}--*"):
        if p.is_dir() and (n := p.name.rsplit("--", 1)[1]).isdigit():
          out.setdefault(int(n), p)
    except OSError:
      pass
  return out


def device_segs(dongle: str, log_id: str) -> dict[int, Path]:
  d = cache.device_dir(dongle, log_id)
  return {int(p.name): p for p in d.iterdir() if p.name.isdigit()} if d.is_dir() else {}


def head(url: str) -> int | None:
  try:
    n = URLFile(url).get_length_online()
  except Exception:
    return None
  return n if n >= 0 else None


class Resolver:
  def __init__(self, dongle: str, log_id: str, sources: list[str], jwt: str | None, sizes: bool):
    self.dongle, self.log_id, self.sources, self.jwt, self.sizes = dongle, log_id, sources, jwt, sizes
    self.errors: dict[str, str] = {}
    self._api = None
    self._local = local_dirs(log_id) if "local" in sources else {}
    self._device = device_segs(dongle, log_id) if "device" in sources else {}
    if "internal" in sources and not filereader.internal_source_available(filereader.DATA_ENDPOINT):
      self.sources = [s for s in sources if s != "internal"]
      self.errors["internal"] = "not reachable"

  def api(self) -> dict:
    if self._api is None:
      try:
        self._api = api_files(self.dongle, self.log_id, self.jwt)
      except Exception as e:
        self._api = {}
        self.errors["api"] = f"{type(e).__name__}: {e}"
    return self._api

  def find(self, source: str, key: Key) -> tuple[str, int | None] | None:
    """(url or local path, size or None) where source has key."""
    if source == "device":
      p = cache.device_path(key)
      return (str(p), p.stat().st_size) if p.is_file() else None
    if source == "local":
      d = self._local.get(key.seg)
      return (str(d / key.file), (d / key.file).stat().st_size) if d and (d / key.file).is_file() else None
    if source == "api":
      hit = self.api().get(key.seg, {}).get(key.file)
      if hit and self.sizes and hit[1] is None:
        hit = (hit[0], head(hit[0]))
      return hit
    if source in ("internal", "openpilotci"):
      url = cache.canonical_url(source, key)
      return (url, n) if (n := head(url)) is not None else None
    return None

  def resolve(self, seg: int, names: tuple[str, ...]) -> dict | None:
    shared = [s for s in self.sources if s in cache.SHARED]
    for name in names:  # cached copies first, from any source this request allows that has the full files
      key = Key(self.dongle, self.log_id, seg, name)
      if (hit := cache.lookup(key, shared)):
        return entry(hit[0], key, hit[1].stat().st_size, True)
    for source in self.sources:
      for name in names:
        key = Key(self.dongle, self.log_id, seg, name)
        try:
          found = self.find(source, key)
        except Exception as e:
          self.errors[source] = f"{type(e).__name__}: {e}"
          found = None
        if found:
          return entry(source, key, found[1], source in ("device", "local"))
    return None


def entry(source: str, key: Key, size: int | None, cached: bool) -> dict:
  return {"url": f"/local/file/{source}/{key}?{TOKEN_PARAM}={config.token}", "file": key.file, "size": size,
          "source": source, "cached": cached}


def jwt_of(req) -> str | None:
  a = req.headers.get("authorization", "")
  return a[4:].strip() if a.startswith("JWT ") else None


def resolve(spec: dict, jwt: str | None = None) -> dict:
  """spec: {route, segs?, kinds?, sources?, sizes?} → {route, segments: {n: {kind: entry}}, missing, errors}"""
  dongle, log_id = parse_route(spec.get("route", ""))
  sources = spec.get("sources") or "auto"
  sources = SOURCES if sources == "auto" else [s for s in sources if s in SOURCES]
  kinds = spec.get("kinds") or ["log"]
  if any(k not in KINDS and k != "log" for k in kinds):
    raise HTTPError(400, "bad kind")
  segs = segments(spec.get("segs"), dongle, log_id)
  r = Resolver(dongle, log_id, sources, jwt, bool(spec.get("sizes")))

  def one(seg, kind):
    if kind == "log":  # the rlog, else the qlog
      return r.resolve(seg, KINDS["rlog"]) or r.resolve(seg, KINDS["qlog"])
    return r.resolve(seg, KINDS[kind])

  jobs = {(seg, kind): _pool.submit(one, seg, kind) for seg in segs for kind in kinds}
  out: dict = {"route": f"{dongle}|{log_id}", "segments": {}, "missing": {}, "errors": r.errors}
  for (seg, kind), fut in jobs.items():
    if (e := fut.result()):
      out["segments"].setdefault(str(seg), {})[kind] = e
    else:
      out["missing"].setdefault(str(seg), []).append(kind)
  return out


@route("POST", "/local/sources/resolve")
def resolve_route(req):
  return resolve(req.json() or {}, jwt_of(req))


def parse_key(req) -> tuple[str, Key]:
  p = req.params
  if p["source"] not in SOURCES or not re.fullmatch(RE.DONGLE_ID, p["dongle"]) or not re.fullmatch(RE.LOG_ID, p["log_id"]) \
     or not p["seg"].isdigit() or p["file"] not in FILES:
    raise HTTPError(404)
  return p["source"], Key(p["dongle"], p["log_id"], int(p["seg"]), p["file"])


def upstream(source: str, key: Key, jwt: str | None) -> str | None:
  """key's URL at source now: the API's signed URLs expire, so they're asked for (API_TTL cached) when a download starts"""
  found = Resolver(key.dongle, key.log_id, [source], jwt, False).find(source, key)
  return found[0] if found else None


# staleness: a blob can change under the same URL (logs get recompressed server side), and every cache keyed by URL would
# keep serving the old bytes. So a whole-file hit is compared with its source's size at most once a day, in the
# background (the hit never waits), and refetched in place on a mismatch along with the other tools' forms of it.

CHECK_EVERY = 24 * 3600
STALE = {"X-Ycpe-Cache": "stale-refetch"}  # served from the old copy while the new one downloads
_checking: set[str] = set()  # canonical URLs
_refetching: dict[str, int] = {}  # canonical URL → inode of the stale copy, which is what gets the stale header
_check_lock = threading.Lock()


def revalidate(key: Key, source: str, path: Path, jwt: str | None) -> None:
  url = cache.canonical_url(source, key)
  if url is None or path != Path(cache_file_path(url)) or time.time() - cache.checked_at(path) < CHECK_EVERY:  # noqa: TID251
    return  # device pulls, decompressed-only copies, or checked lately
  with _check_lock:
    if url in _checking:
      return
    _checking.add(url)

  def check():
    try:
      found = Resolver(key.dongle, key.log_id, [source], jwt, True).find(source, key)
      if not found or found[1] is None:
        return  # unreachable or gone: keep the copy, ask again on a later hit
      if found[1] == path.stat().st_size:
        cache.mark_checked(path)
        return
      cache.stats["stale"] += 1
      _refetching[url] = path.stat().st_ino
      cache.remove(cache.derived_copies(url))
      fill = cache.fill(key, source, found[0])
      fill.finished.wait()
    except Exception as e:
      print(f"revalidating {key}: {type(e).__name__}: {e}")
    finally:
      _refetching.pop(url, None)
      with _check_lock:
        _checking.discard(url)
  threading.Thread(target=check, daemon=True, name=f"check {key}").start()


@route("GET", "/local/file/{source}/{dongle}/{log_id}/{seg}/{file}")
def serve_file(req):
  """The file's bytes: from the shared cache, a local directory, or downloaded from the source into the cache while
  it streams. A Range request on a file that isn't cached yet is proxied, and the full download starts behind it."""
  source, key = parse_key(req)
  if source == "local":
    found = Resolver(key.dongle, key.log_id, ["local"], None, False).find("local", key)
    if not found:
      raise HTTPError(404)
    return File(str(confine(found[0], config.log_roots)), headers=HIT)
  if (hit := cache.lookup(key)):
    cache.stats["hits"] += 1
    cache.touch(hit[1])
    revalidate(key, *hit, jwt_of(req))
    stale = _refetching.get(cache.canonical_url(hit[0], key) or "") == hit[1].stat().st_ino
    return File(str(hit[1]), headers=STALE if stale else HIT)
  if source == "device":
    raise HTTPError(404)
  cache.stats["misses"] += 1
  if (fill := cache.filling(key)) is None:
    if (url := upstream(source, key, jwt_of(req))) is None:
      raise HTTPError(404)
    fill = cache.fill(key, source, url)
  rng = req.headers.get("range")
  if rng:
    return proxy_range(fill.url, rng)
  # follow the download: the temporary file is opened here, so its rename into place can't race the response
  while not fill.finished.wait(0.01):
    try:
      fh = open(fill.tmp, "rb")
    except FileNotFoundError:
      continue
    return Stream(file_chunks(fh, 0, fill.total, fill.running), headers={"Accept-Ranges": "bytes", **MISS}, length=fill.total)
  if fill.error:
    raise HTTPError(502, fill.error)
  return File(str(fill.final), headers=MISS)


def proxy_range(url: str, rng: str) -> Stream:
  r = URLFile.pool_manager().request("GET", url, headers={"Range": rng}, preload_content=False, timeout=30.0)
  if r.status not in (200, 206):
    r.release_conn()
    raise HTTPError(502, f"upstream {r.status}")
  headers = {"Accept-Ranges": "bytes", **MISS}
  if (cr := r.headers.get("content-range")):
    headers["Content-Range"] = cr

  def gen():
    try:
      yield from r.stream(cache.CHUNK_SIZE)
    finally:
      r.release_conn()
  n = r.headers.get("content-length")
  return Stream(gen(), "application/octet-stream", headers, status=r.status, length=int(n) if n else None)


def fetch(key: Key, sources=None, jwt: str | None = None, job=None) -> Path:
  """A local path holding key, downloading it into the shared cache first if needed (blocking). With a job, the wait
  stops when it's cancelled, and so does a download this call started."""
  srcs = [s for s in (sources or SOURCES) if s in SOURCES]
  if (hit := cache.lookup(key, [s for s in srcs if s in cache.SHARED])):
    cache.touch(hit[1])
    revalidate(key, *hit, jwt)
    return hit[1]
  if "local" in srcs and (found := Resolver(key.dongle, key.log_id, ["local"], None, False).find("local", key)):
    return confine(found[0], config.log_roots)
  if (fill := cache.filling(key)) is None:
    r = Resolver(key.dongle, key.log_id, [s for s in srcs if s not in ("device", "local")], jwt, False)
    found = next(((s, f) for s in r.sources if (f := r.find(s, key))), None)
    if found is None:
      raise FileNotFoundError(f"{key} isn't in {', '.join(srcs)}" + "".join(f"; {k}: {v}" for k, v in r.errors.items()))
    fill = cache.fill(key, found[0], found[1][0], cancelled=(lambda: job.cancelled) if job is not None else None)
  while not fill.finished.wait(0.1):
    if job is not None:
      job.check()
  if fill.error:
    raise OSError(fill.error)
  return fill.final
