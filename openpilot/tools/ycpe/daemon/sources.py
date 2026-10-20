"""Where a route's files come from, in LogReader's auto order, and a URL that serves them through the shared cache.

Order: device (pulled over ssh) → local (log dirs on this machine) → internal (DATA_ENDPOINT) → comma API →
openpilotci → commaCarSegments, the same as logreader.auto_source after the local ones. A copy already in the
shared cache wins over any download.
"""
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from openpilot.tools.lib.api import CommaApi
from openpilot.tools.lib.auth_config import get_token
from openpilot.tools.lib.filereader import DATA_ENDPOINT, internal_source_available
from openpilot.tools.lib.helpers import RE
from openpilot.tools.lib.url_file import URLFile
from openpilot.tools.ycpe.daemon import cache
from openpilot.tools.ycpe.daemon.cache import Key
from openpilot.tools.ycpe.daemon.core import TOKEN_PARAM, File, HTTPError, Stream, config, confine, file_chunks, module, route

SOURCES = ["device", "local", "internal", "api", "openpilotci", "carSegments"]
KINDS = {
  "rlog": ("rlog.zst", "rlog.bz2"), "qlog": ("qlog.zst", "qlog.bz2"), "qcamera": ("qcamera.ts",),
  "fcamera": ("fcamera.hevc",), "ecamera": ("ecamera.hevc",), "dcamera": ("dcamera.hevc",),
}
FILES = {f: k for k, fs in KINDS.items() for f in fs}
API_KINDS = {"logs": "rlog", "qlogs": "qlog", "cameras": "fcamera", "ecameras": "ecamera", "dcameras": "dcamera", "qcameras": "qcamera"}
ROUTE_RE = re.compile(RE.SEGMENT_RANGE)
HIT, MISS = {"X-Ycpe-Cache": "hit"}, {"X-Ycpe-Cache": "miss"}  # whether bytes came off local disk, for the page's status
API_TTL = 30 * 60  # signed URLs from the route listing are reused this long

module("sources", "1", sources=SOURCES, kinds=[*KINDS, "log"])
_pool = ThreadPoolExecutor(16, thread_name_prefix="probe")


def parse_route(text: str) -> tuple[str, str, str, str | None]:
  """(dongle, log_id, slice, selector) of "dongle|log_id[/slice][/selector]" (LogReader's syntax; | / _ separators)."""
  m = ROUTE_RE.fullmatch(text.strip())
  if not m:
    raise HTTPError(400, "bad route")
  return m.group("dongle_id"), m.group("log_id"), m.group("slice") or "", m.group("selector")


def pick(seg_count: int, sl: str) -> list[int]:
  m = re.fullmatch(RE.SLICE, sl)
  start, end, step = (None if s is None else int(s) for s in m.groups())
  segs = list(range(seg_count))
  if start is not None and end is None and ":" not in sl:
    return [segs[start]] if -seg_count <= start < seg_count else []
  return segs[slice(start, end, step)]


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
  d = cache.root() / "ycpe" / "device" / dongle / log_id
  return {int(p.name): p for p in d.iterdir() if p.name.isdigit()} if d.is_dir() else {}


def head(url: str) -> int | None:
  try:
    n = URLFile(url).get_length_online()
  except Exception:
    return None
  return n if n >= 0 else None


def car_segments_url(key: Key) -> str | None:
  from openpilot.tools.lib.comma_car_segments import get_url
  return get_url(key.route, key.seg, key.file)


class Resolver:
  def __init__(self, dongle: str, log_id: str, sources: list[str], jwt: str | None, sizes: bool):
    self.dongle, self.log_id, self.sources, self.jwt, self.sizes = dongle, log_id, sources, jwt, sizes
    self.errors: dict[str, str] = {}
    self._api = None
    self._local = local_dirs(log_id) if "local" in sources else {}
    self._device = device_segs(dongle, log_id) if "device" in sources else {}
    if "internal" in sources and not internal_source_available(DATA_ENDPOINT):
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

  def known_segments(self) -> int:
    segs = set(self._local) | set(self._device)
    if "api" in self.sources:
      segs |= set(self.api())
    return max(segs) + 1 if segs else 0

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
    if source == "carSegments" and key.file == "rlog.zst":
      try:
        url = car_segments_url(key)
      except Exception:
        return None
      return (url, head(url)) if url else None
    return None

  def resolve(self, seg: int, names: tuple[str, ...]) -> dict | None:
    for name in names:  # cached copies first, from any source this request allows
      key = Key(self.dongle, self.log_id, seg, name)
      if (hit := cache.lookup(key, [s for s in self.sources if s != "local"])):
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
          remember(source, key, found[0])
          return entry(source, key, found[1], source in ("device", "local"))
    return None


_upstream: dict[tuple[str, Key], str] = {}


def remember(source: str, key: Key, url: str) -> None:
  if source not in ("device", "local"):
    _upstream[(source, key)] = url


def entry(source: str, key: Key, size: int | None, cached: bool) -> dict:
  return {"url": f"/local/file/{source}/{key}?{TOKEN_PARAM}={config.token}", "file": key.file, "size": size,
          "source": source, "cached": cached}


def jwt_of(req) -> str | None:
  a = req.headers.get("authorization", "")
  return a[4:].strip() if a.startswith("JWT ") else None


def resolve(spec: dict, jwt: str | None = None) -> dict:
  """spec: {route, segs?, kinds?, sources?, mode?, sizes?} → {route, segments: {n: {kind: entry}}, missing, errors}"""
  dongle, log_id, sl, selector = parse_route(str(spec.get("route", "")))
  mode = spec.get("mode") or selector or "a"
  if mode not in ("a", "r", "q"):
    raise HTTPError(400, "bad mode")
  sources = spec.get("sources") or "auto"
  sources = SOURCES if sources == "auto" else [s for s in sources if s in SOURCES]
  kinds = spec.get("kinds") or ["log"]
  if any(k not in KINDS and k != "log" for k in kinds):
    raise HTTPError(400, "bad kind")
  r = Resolver(dongle, log_id, sources, jwt, bool(spec.get("sizes")))
  segs = spec.get("segs")
  if segs is None:
    segs = pick(r.known_segments(), sl)
  elif not all(isinstance(n, int) and 0 <= n < 10000 for n in segs):
    raise HTTPError(400, "bad segs")

  def one(seg, kind):
    if kind != "log":
      return r.resolve(seg, KINDS[kind])
    order = {"r": ["rlog"], "q": ["qlog"], "a": ["rlog", "qlog"]}[mode]
    for k in order:
      if (e := r.resolve(seg, KINDS[k])):
        return e
    return None

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
  if (url := _upstream.get((source, key))):
    return url
  found = Resolver(key.dongle, key.log_id, [source], jwt, False).find(source, key)
  if found:
    remember(source, key, found[0])
  return found[0] if found else None


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
    return File(str(hit[1]), headers=HIT)
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
  if (hit := cache.lookup(key)):
    cache.touch(hit[1])
    return hit[1]
  srcs = [s for s in (sources or SOURCES) if s in SOURCES]
  if "local" in srcs and (found := Resolver(key.dongle, key.log_id, ["local"], None, False).find("local", key)):
    return Path(found[0])
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
