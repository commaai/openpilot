"""The download cache shared with openpilot's tools (Paths.download_cache_root()).

Two formats already live there, both keyed by the URL without its query, so signed URLs of one file share a key:
- file_downloader.py (cabana, jotpluggler, replay): sha256(url) holds the whole file, and
  sha256("decompressed-<zst|bz2>:" + url) a decompressed log.
- url_file.py (LogReader, FrameReader, process_replay): md5(url)_<n>.0 in 1 MB chunks, plus md5(url)_length.
A file is looked up under the canonical URL of every source that could hold it (internal, comma API blob,
openpilotci, commaCarSegments). Downloads land as sha256(url), and logs are also mirrored into url_file chunks,
so one fetch serves all of them. Files the daemon makes itself (device pulls, transcodes) live under ycpe/.
The whole directory is held under a budget by LRU on mtime; reads bump mtime.
"""
import hashlib
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from openpilot.tools.lib.filereader import DATA_ENDPOINT
from openpilot.tools.lib.openpilotci import get_url as openpilotci_url
from openpilot.tools.lib.url_file import CHUNK_SIZE, URLFile, hash_url
from openpilot.tools.ycpe.daemon.core import HTTPError, config, module, route

API_BLOB = "https://commadata2.blob.core.windows.net"
CAR_SEGMENTS = "https://huggingface.co/datasets/commaai/commaCarSegments/resolve/main/segments"
LOGS = ("rlog.zst", "rlog.bz2", "qlog.zst", "qlog.bz2")
TMP = "ycpe-tmp-"
PREFETCH = 8  # downloads at once: one Azure blob stream gives ~5 MB/s here
RECENT = 120  # seconds: files written or read this recently are never pruned (they may be in use)


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
  return Path(config.cache_root)


def canonical_url(source: str, key: Key) -> str | None:
  """The query-less URL the shared cache knows this file by when it came from source."""
  path = f"{key.dongle}/{key.log_id}/{key.seg}/{key.file}"
  if source == "internal":
    return f"{DATA_ENDPOINT.rstrip('/')}/{path}"
  if source == "api":
    container = "qlog" if key.file.startswith(("qlog", "qcamera")) else "commadata2"
    return f"{API_BLOB}/{container}/{path}"
  if source == "openpilotci":
    return openpilotci_url(key.route, key.seg, key.file)
  if source == "carSegments":
    return f"{CAR_SEGMENTS}/{path}"
  return None


def sha(url: str) -> str:
  return hashlib.sha256(url.split("?", 1)[0].encode()).hexdigest()


def derived(name: str) -> Path:
  """A path under ycpe/ for files the daemon makes (device pulls, transcodes, replays); counted in the budget."""
  p = (root() / "ycpe" / name).resolve()
  if not str(p).startswith(str((root() / "ycpe").resolve()) + os.sep):
    raise HTTPError(403, "path traversal")
  p.parent.mkdir(parents=True, exist_ok=True)
  return p


def device_path(key: Key) -> Path:
  """Where a log pulled from a device lives (fields of a Key are validated by the caller)."""
  return root() / "ycpe" / "device" / str(key)


def touch(path: Path) -> None:
  try:
    os.utime(path)
  except OSError:
    pass


def _chunks_complete(url: str) -> int | None:
  h = hash_url(url)
  try:
    length = int((root() / f"{h}_length").read_text())
  except (OSError, ValueError):
    return None
  n = -(-length // CHUNK_SIZE)
  return length if length > 0 and all((root() / f"{h}_{float(i)}").is_file() for i in range(n)) else None


def lookup(key: Key, sources=("device", "internal", "api", "openpilotci", "carSegments")) -> tuple[str, Path] | None:
  """(source, path) of a cached copy of key, or None. A complete url_file chunk set is joined into a whole file
  first (which cabana then reuses too). A decompressed-only copy is served as is; the page's reader takes raw logs."""
  if "device" in sources and (p := device_path(key)).is_file():
    return "device", p
  for source in sources:
    if (url := canonical_url(source, key)) is None:
      continue
    if (p := root() / sha(url)).is_file():
      return source, p
    if key.file in LOGS and (length := _chunks_complete(url)):
      tmp = root() / f"{TMP}{sha(url)}"
      with open(tmp, "wb") as out:
        h = hash_url(url)
        for i in range(-(-length // CHUNK_SIZE)):
          out.write((root() / f"{h}_{float(i)}").read_bytes())
      os.replace(tmp, p)
      return source, p
    for comp in ("zst", "bz2"):
      if (p := root() / sha(f"decompressed-{comp}:{url}")).is_file():
        return source, p
  return None


def mirror_chunks(url: str, path: Path) -> None:
  """Writes a downloaded log as url_file chunks too, so LogReader and process_replay find it."""
  h, size = hash_url(url), path.stat().st_size
  names = []
  with open(path, "rb") as f:
    for i in range(-(-size // CHUNK_SIZE)):
      name = f"{h}_{float(i)}"
      tmp = root() / f"{TMP}{name}"
      tmp.write_bytes(f.read(CHUNK_SIZE))
      os.replace(tmp, root() / name)
      names.append(name)
  tmp = root() / f"{TMP}{h}_length"
  tmp.write_text(str(size))
  os.replace(tmp, root() / f"{h}_length")
  # url_file prunes by its own manifest; list the chunks there so its accounting sees them
  manifest = root() / "manifest.txt"
  with _manifest_lock:
    lines = manifest.read_text().splitlines() if manifest.exists() else []
    now = int(time.time())  # noqa: TID251  url_file's manifest uses wall clock
    tmp = root() / f"{TMP}manifest"
    tmp.write_text("\n".join(lines + [f"{n} {now}" for n in names]))
    os.replace(tmp, manifest)


_manifest_lock = threading.Lock()


# downloads: one per key at a time, readable while they run

class Fill:
  """Downloads url into the cache as sha256(cache_url); readers can follow tmp while it grows."""
  def __init__(self, key: Key, source: str, url: str, cache_url: str):
    self.key, self.source, self.url, self.cache_url = key, source, url, cache_url
    self.final = root() / sha(cache_url)
    self.tmp = root() / f"{TMP}{sha(cache_url)}.{threading.get_ident()}.{time.monotonic_ns()}"
    self.done, self.total = 0, None
    self.error: str | None = None
    self.finished = threading.Event()
    self.started = time.monotonic()

  def running(self) -> bool:
    if self.error:
      raise OSError(self.error)
    return not self.finished.is_set()

  def run(self, cancelled=lambda: False) -> None:
    try:
      root().mkdir(parents=True, exist_ok=True)
      r = URLFile.pool_manager().request("GET", self.url, preload_content=False, timeout=30.0)
      try:
        if r.status != 200:
          raise OSError(f"HTTP {r.status} from {self.source}")
        self.total = int(r.headers.get("content-length") or 0) or None
        with open(self.tmp, "wb") as f:
          for data in r.stream(CHUNK_SIZE):
            if cancelled():
              raise OSError("cancelled")
            f.write(data)
            f.flush()
            self.done += len(data)
      finally:
        r.release_conn()
      if self.total is not None and self.done != self.total:
        raise OSError(f"short read {self.done}/{self.total}")
      os.replace(self.tmp, self.final)
      stats["fetched"] += self.done
      if self.key.file in LOGS:
        mirror_chunks(self.cache_url, self.final)
    except Exception as e:
      self.error = f"{type(e).__name__}: {e}"
      try:
        os.unlink(self.tmp)
      except OSError:
        pass
    finally:
      with _fills_lock:
        _fills.pop(self.key, None)
      self.finished.set()
      stale()
      prune_soon()


_fills: dict[Key, Fill] = {}
_fills_lock = threading.Lock()
stats = {"hits": 0, "misses": 0, "served": 0, "fetched": 0}


def fill(key: Key, source: str, url: str, cancelled=None) -> Fill:
  """The download of key in flight, starting one in a thread if there's none. cancelled() can abort a new one."""
  with _fills_lock:
    if (f := _fills.get(key)) is not None:
      return f
    f = _fills[key] = Fill(key, source, url, canonical_url(source, key) or url)
  threading.Thread(target=f.run, args=(cancelled or (lambda: False),), daemon=True, name=f"fill {key}").start()
  return f


def filling(key: Key) -> Fill | None:
  return _fills.get(key)


# budget

_prune_lock = threading.Lock()
_prune_timer: threading.Timer | None = None


def scan() -> list[tuple[float, int, Path]]:
  out = []
  for dirpath, _, names in os.walk(root()):
    for n in names:
      p = Path(dirpath) / n
      try:
        st = p.stat()
      except OSError:
        continue
      out.append((st.st_mtime, st.st_size, p))
  return out


def prune(budget: int | None = None) -> dict:
  """Deletes least recently used files until the cache fits the budget. Skips in-flight and recent files."""
  budget = config.cache_budget if budget is None else budget
  with _prune_lock:
    files = sorted(scan())
    total = sum(s for _, s, _ in files)
    removed = freed = 0
    now = time.time()  # noqa: TID251  compared with file mtimes
    for mtime, size, p in files:
      if total <= budget:
        break
      if p.name.startswith(TMP) or p.name == "manifest.txt" or now - mtime < RECENT:
        continue
      try:
        p.unlink()
      except OSError:
        continue
      total -= size
      freed += size
      removed += 1
    stale()
    return {"removed": removed, "freed": freed, "bytes": total}


def prune_soon() -> None:
  global _prune_timer
  if _prune_timer is None or not _prune_timer.is_alive():
    _prune_timer = threading.Timer(2.0, prune)
    _prune_timer.daemon = True
    _prune_timer.start()


def evict(key: Key) -> int:
  """Removes every cached copy of key; returns bytes freed."""
  freed = 0
  paths = [device_path(key)]
  for source in ("internal", "api", "openpilotci", "carSegments"):
    url = canonical_url(source, key)
    h = hash_url(url)
    paths += [root() / sha(url), root() / sha(f"decompressed-zst:{url}"), root() / sha(f"decompressed-bz2:{url}"), root() / f"{h}_length"]
    paths += list(root().glob(f"{h}_*.0"))
  for p in paths:
    try:
      freed += p.stat().st_size
      p.unlink()
    except OSError:
      pass
  stale()
  return freed


def fetch(dongle: str, log_id: str, seg: int, file: str, sources=None, job=None) -> Path:
  """For other modules: a local path holding the file, downloaded into the shared cache if needed (blocking)."""
  from openpilot.tools.ycpe.daemon.sources import fetch as fetch_key
  return fetch_key(Key(dongle, log_id, seg, file), sources, None, job)


# endpoints (sources.py resolves; it imports this module, so it's imported lazily here)

module("cache", "1")


_usage = (0.0, 0, 0)  # (monotonic time, bytes, files): the page polls this, so the directory walk is reused for 2 s


def stale() -> None:
  global _usage
  _usage = (0.0, 0, 0)


def usage() -> dict:
  global _usage
  if time.monotonic() - _usage[0] > 2:
    files = scan()
    _usage = (time.monotonic(), sum(s for _, s, _ in files), len(files))
  return {"dir": str(root()), "budget": config.cache_budget, "bytes": _usage[1], "files": _usage[2],
          "fills": [{"key": str(f.key), "source": f.source, "done": f.done, "total": f.total} for f in list(_fills.values())], **stats}


@route("GET", "/local/cache")
def cache_stats(req):
  """Usage of the whole shared cache. With ?route=dongle|log_id[&segs=N], also which of its files are cached."""
  from openpilot.tools.ycpe.daemon.sources import KINDS, Resolver, parse_route
  out = usage()
  if (r := req.query.get("route")):
    dongle, log_id, _, _ = parse_route(r)
    count = req.arg("segs", None, int) or Resolver(dongle, log_id, ["device", "local", "api"], None, False).known_segments()
    segs = {}
    for seg in range(min(count, 10000)):
      have = {}
      for kind, names in KINDS.items():
        for name in names:
          if (hit := lookup(Key(dongle, log_id, seg, name))):
            have[kind] = {"file": name, "source": hit[0], "size": hit[1].stat().st_size}
            break
      segs[str(seg)] = have
    out["route"] = {"name": f"{dongle}|{log_id}", "segments": segs}
  return out


@route("POST", "/local/cache/prefetch")
def cache_prefetch(req):
  """Body as /local/sources/resolve. Downloads every resolved file that isn't cached yet, PREFETCH at a time, as a job whose
  meta carries the route and each segment's state (queued, loading, disk, error) for the page's load lanes."""
  from concurrent.futures import ThreadPoolExecutor
  from openpilot.tools.ycpe.daemon import jobs
  from openpilot.tools.ycpe.daemon.sources import fetch as fetch_key, jwt_of, resolve
  jwt = jwt_of(req)
  res = resolve({**(req.json() or {}), "sizes": True}, jwt)
  dongle, log_id = res["route"].split("|")
  items = [(Key(dongle, log_id, int(n), e["file"]), e) for n, kinds in res["segments"].items() for e in kinds.values()]
  state = {key: "disk" if e["cached"] else "queued" for key, e in items}
  todo = [(key, e) for key, e in items if not e["cached"]]
  total = sum(e["size"] or 0 for _, e in todo)

  def segs() -> dict:
    out: dict[str, str] = {}
    for key, s in state.items():
      cur = out.get(str(key.seg), "disk")
      out[str(key.seg)] = min(cur, s, key=["error", "loading", "queued", "disk"].index)
    return out

  def one(key, e, job):
    state[key] = "loading"
    try:
      fetch_key(key, [e["source"]], jwt, job)
      state[key] = "disk"
    except jobs.Cancelled:
      state[key] = "queued"
      raise
    except Exception:
      state[key] = "error"
      raise

  def run(job):
    t0 = time.monotonic()
    job.progress(0, total, unit="bytes")
    with ThreadPoolExecutor(PREFETCH) as ex:
      futs = [ex.submit(one, key, e, job) for key, e in todo]
      while not all(f.done() for f in futs):
        time.sleep(0.2)
        if job.cancelled:
          ex.shutdown(cancel_futures=True)
          job.check()
        done = sum(e["size"] or 0 for key, e in todo if state[key] == "disk")
        done += sum(f.done for f in list(_fills.values()) if f.key.route == res["route"])
        dt = time.monotonic() - t0
        job.progress(done, total, text=f"{done / dt / 1e6:.0f} MB/s", segs=segs())
    dt = time.monotonic() - t0
    errors = [f.exception() for f in futs if f.exception()]
    job.progress(total, total, text=f"{total / dt / 1e6:.0f} MB/s" if dt > 0 else "", segs=segs())
    if errors:
      raise OSError(f"{len(errors)} of {len(todo)} files failed, first: {errors[0]}")
    return {"files": len(todo), "bytes": total, "seconds": round(dt, 2)}

  return jobs.start("prefetch", run, title=f"cache {res['route']}", route=res["route"], segs=segs()).json()


@route("POST", "/local/cache/evict")
def cache_evict(req):
  """Body {route, segs?, kinds?}: deletes those files' copies in the shared cache (never anything remote)."""
  from openpilot.tools.ycpe.daemon.sources import KINDS, Resolver, parse_route
  body = req.json() or {}
  dongle, log_id, _, _ = parse_route(str(body.get("route", "")))
  kinds = [k for k in (body.get("kinds") or KINDS) if k in KINDS]
  segs = body.get("segs")
  if segs is None:
    segs = range(Resolver(dongle, log_id, ["device", "local", "api"], None, False).known_segments() or 1000)
  freed = sum(evict(Key(dongle, log_id, int(n), name)) for n in segs for kind in kinds for name in KINDS[kind])
  return {"freed": freed, **usage()}


@route("POST", "/local/cache/prune")
def cache_prune(req):
  """Body {budget?}: sets the budget in bytes (for this run) and prunes to it."""
  body = req.json() or {}
  if body.get("budget"):
    config.cache_budget = int(body["budget"])
  return {**prune(), "budget": config.cache_budget}
