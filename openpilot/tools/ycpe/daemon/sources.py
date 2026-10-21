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
from openpilot.tools.ycpe.daemon import cache, jobs
from openpilot.tools.ycpe.daemon.cache import Key
from openpilot.tools.ycpe.daemon.core import TOKEN_PARAM, File, HTTPError, Stream, config, confine, file_chunks, route

SOURCES = ["device", "local", "internal", "api", "openpilotci"]
KINDS = {"rlog": FileName.RLOG, "qlog": FileName.QLOG, "qcamera": FileName.QCAMERA, "fcamera": FileName.FCAMERA,
         "ecamera": FileName.ECAMERA, "dcamera": FileName.DCAMERA}
FILES = {f: k for k, fs in KINDS.items() for f in fs}
HIT, MISS = {"X-Ycpe-Cache": "hit"}, {"X-Ycpe-Cache": "miss"}  # whether bytes came off local disk, for the page's status
API_TTL = 30 * 60  # signed URLs from the route listing are reused this long
PREFETCH = 8  # downloads at once: one Azure blob stream gives ~5 MB/s here

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

  futs = {(seg, kind): _pool.submit(one, seg, kind) for seg in segs for kind in kinds}
  out: dict = {"route": f"{dongle}|{log_id}", "segments": {}, "missing": {}, "errors": r.errors}
  for (seg, kind), fut in futs.items():
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


def download(key: Key, source: str, jwt: str | None, cancelled=None) -> cache.Fill:
  """key's download into the shared cache: the one in flight, else one from source started now (the API's signed URLs
  expire, so they're asked for, API_TTL cached, at this point). 404 if source doesn't have key."""
  if (fill := cache.filling(key)) is None:
    if not (found := Resolver(key.dongle, key.log_id, [source], jwt, False).find(source, key)):
      raise HTTPError(404, f"no {key} in {source}")
    fill = cache.fill(key, source, found[0], cancelled)
  return fill


# staleness: a blob can change under the same URL (logs get recompressed server side), and every cache keyed by URL would
# keep serving the old bytes. So a whole-file hit is compared with its source's size at most once a day, in the
# background (the hit never waits), and refetched in place on a mismatch along with the other tools' forms of it.

CHECK_EVERY = 24 * 3600
_checking: set[str] = set()  # canonical URLs
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
      cache.remove(cache.derived_copies(url))
      fill = cache.fill(key, source, found[0])
      fill.finished.wait()
    except Exception as e:
      print(f"revalidating {key}: {type(e).__name__}: {e}")
    finally:
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
    cache.touch(hit[1])
    revalidate(key, *hit, jwt_of(req))
    return File(str(hit[1]), headers=HIT)  # the old copy while a stale one is refetched
  if source == "device":
    raise HTTPError(404)
  fill = download(key, source, jwt_of(req))
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


# the shared cache, by route

@route("GET", "/local/cache")
def cache_stats(req):
  """Usage of the whole shared cache. With ?route=dongle|log_id[&segs=N], also which of its files are cached."""
  out = cache.usage()
  if (r := req.query.get("route")):
    dongle, log_id = parse_route(r)
    count = req.arg("segs", None, int)
    segs = {}
    for seg in segments(None, dongle, log_id) if count is None else range(min(count, 10000)):
      have = segs[str(seg)] = {}
      for kind, names in KINDS.items():
        for name in names:
          if (hit := cache.lookup(Key(dongle, log_id, seg, name))):
            have[kind] = {"file": name, "source": hit[0], "size": hit[1].stat().st_size}
            break
    out["route"] = {"name": f"{dongle}|{log_id}", "segments": segs}
  return out


def fetch(key: Key, source: str, jwt: str | None, job, moved) -> None:
  """Downloads key from source into the shared cache unless a copy is there, telling moved(bytes in so far), until
  it's in or job is cancelled (which stops a download this call started too)."""
  if cache.lookup(key):
    return
  fill = download(key, source, jwt, lambda: job.cancelled)
  while not fill.finished.wait(0.2):
    moved(fill.done)
    job.check()
  if fill.error:
    raise OSError(fill.error)


@route("POST", "/local/cache/prefetch")
def cache_prefetch(req):
  """Body as /local/sources/resolve. Downloads every resolved file that isn't cached yet, PREFETCH at a time, as a job whose
  meta carries the route and each segment's state (queued, loading, disk, error) for the page's load lanes."""
  jwt = jwt_of(req)
  res = resolve({**(req.json() or {}), "sizes": True}, jwt)
  dongle, log_id = res["route"].split("|")
  found = {Key(dongle, log_id, int(n), e["file"]): e for n, kinds in res["segments"].items() for e in kinds.values()}
  todo = [(key, e["size"] or 0) for key, e in found.items() if not e["cached"]]

  def run(job):
    def one(key, moved):
      fetch(key, found[key]["source"], jwt, job, moved)
    return jobs.run_files(job, todo, one, PREFETCH, have=[key for key, e in found.items() if e["cached"]])
  return jobs.start("prefetch", run, title=f"cache {res['route']}", route=res["route"]).json()


@route("POST", "/local/cache/evict")
def cache_evict(req):
  """Body {route, segs?, kinds?}: deletes those files' copies in the shared cache (never anything remote)."""
  body = req.json() or {}
  dongle, log_id = parse_route(body.get("route", ""))
  kinds = [k for k in (body.get("kinds") or KINDS) if k in KINDS]
  segs = segments(body.get("segs"), dongle, log_id)
  freed = sum(cache.evict(Key(dongle, log_id, n, name)) for n in segs for kind in kinds for name in KINDS[kind])
  return {"freed": freed, **cache.usage()}
