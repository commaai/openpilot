"""The download cache shared with openpilot's tools (Paths.download_cache_root()).

Two formats already live there, both keyed by the URL without its query, so signed URLs of one file share a key:
- file_downloader.py (cabana, jotpluggler, replay): sha256(url) holds the whole file, and
  sha256("decompressed-<zst|bz2>:" + url) a decompressed log.
- url_file.py (LogReader, FrameReader, process_replay): md5(url)_<n>.0 in 1 MB chunks, plus md5(url)_length.
A file is looked up under the canonical URL of every source that could hold it (internal, comma API blob,
openpilotci). Downloads land as sha256(url), and logs are also mirrored into url_file chunks,
so one fetch serves all of them. Files the daemon makes itself (device pulls, transcodes) live under ycpe/.
The whole directory is held under a budget by LRU on mtime; reads bump mtime.
"""
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from openpilot.common.hardware.hw import Paths
from openpilot.tools.lib import filereader
from openpilot.tools.lib.azure_container import AzureContainer
from openpilot.tools.lib.file_downloader import cache_file_path
from openpilot.tools.lib.openpilotci import get_url as openpilotci_url
from openpilot.tools.lib.openpilotcontainers import DataProdContainer
from openpilot.tools.lib.route import FileName
from openpilot.tools.lib.url_file import CHUNK_SIZE, URLFile, hash_url
from openpilot.tools.ycpe.daemon.core import HTTPError, config, module, route

QLOGS = AzureContainer("commadata2", "qlog")  # where the API keeps qlogs and qcameras; the rest is in DataProdContainer
# device, internal, api and openpilotci hold the same bytes for a file, so a copy from any of them stands in for the others
# (and local directories hold them too)
SHARED = ("device", "internal", "api", "openpilotci")
LOGS = FileName.RLOG + FileName.QLOG
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


def ycpe_dir(name: str) -> Path:
  """The directory under ycpe/ for one kind of file the daemon makes (device pulls, transcodes); counted in the budget."""
  return root() / "ycpe" / name


def device_dir(dongle: str, log_id: str) -> Path:
  """Where a route's logs pulled from a device live, one directory per segment (the caller validates the names)."""
  return ycpe_dir("device") / dongle / log_id


def device_path(key: Key) -> Path:
  return device_dir(key.dongle, key.log_id) / str(key.seg) / key.file


def touch(path: Path) -> None:
  try:
    os.utime(path)
  except OSError:
    pass


def chunk_paths(url: str) -> tuple[int, list[Path]] | None:
  """url_file's chunk set of url, as its _length file says: (length, the chunks' paths), or None without one."""
  h = hash_url(url)
  try:
    length = int((root() / f"{h}_length").read_text())
  except (OSError, ValueError):
    return None
  return length, [root() / f"{h}_{float(i)}" for i in range(-(-length // CHUNK_SIZE))]


def lookup(key: Key, sources=SHARED) -> tuple[str, Path] | None:
  """(source, path) of a cached copy of key from one of sources, or None. A complete url_file chunk set is joined into a whole file
  first (which cabana then reuses too). A decompressed-only copy is served as is; the page's reader takes raw logs."""
  if "device" in sources and (p := device_path(key)).is_file():
    return "device", p
  for source in sources:
    if (url := canonical_url(source, key)) is None:
      continue
    if (p := Path(cache_file_path(url))).is_file():
      return source, p
    if key.file in LOGS and (c := chunk_paths(url)) and c[0] > 0 and all(chunk.is_file() for chunk in c[1]):
      length, chunks = c
      tmp = root() / f"{TMP}{p.name}"
      with open(tmp, "wb") as out:
        for chunk in chunks:
          out.write(chunk.read_bytes())
      os.replace(tmp, p)
      account(length, 1)
      return source, p
    for comp in ("zst", "bz2"):
      if (p := Path(cache_file_path(url, comp))).is_file():
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
    old = manifest.stat().st_size if manifest.exists() else None
    lines = manifest.read_text().splitlines() if old is not None else []
    now = int(time.time())  # noqa: TID251  url_file's manifest uses wall clock
    tmp = root() / f"{TMP}manifest"
    tmp.write_text("\n".join(lines + [f"{n} {now}" for n in names]))
    os.replace(tmp, manifest)
    account(size + len(str(size)) + manifest.stat().st_size - (old or 0), len(names) + 1 + (old is None))


_manifest_lock = threading.Lock()


# usage: a running total, kept by downloads, evictions and prunes, and corrected by a full walk at most every RESCAN
# seconds (other tools write here too). The page polls it, so reading it must not walk the directory.

RESCAN = 60
_usage = {"at": -RESCAN, "bytes": 0, "files": 0}
_usage_lock = threading.Lock()


def account(nbytes: int, nfiles: int) -> None:
  with _usage_lock:
    _usage["bytes"] += nbytes
    _usage["files"] += nfiles


def totals() -> tuple[int, int]:
  """(bytes, files) in the cache directory."""
  if time.monotonic() - _usage["at"] > RESCAN:
    files = scan()
    with _usage_lock:
      _usage.update(at=time.monotonic(), bytes=sum(s for _, s, _ in files), files=len(files))
  return _usage["bytes"], _usage["files"]


# downloads: one per key at a time, readable while they run

class Fill:
  """Downloads url into the cache as sha256(cache_url); readers can follow tmp while it grows."""
  def __init__(self, key: Key, source: str, url: str, cache_url: str):
    self.key, self.source, self.url, self.cache_url = key, source, url, cache_url
    self.final = Path(cache_file_path(cache_url))
    self.tmp = root() / f"{TMP}{self.final.name}.{threading.get_ident()}.{time.monotonic_ns()}"
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
      old = self.final.stat().st_size if self.final.exists() else None  # a stale copy being replaced
      os.replace(self.tmp, self.final)
      mark_checked(self.final)
      stats["fetched"] += self.done
      account(self.done - (old or 0), 0 if old is not None else 1)
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
      prune_soon()


_fills: dict[Key, Fill] = {}
_fills_lock = threading.Lock()
stats = {"hits": 0, "misses": 0, "fetched": 0, "stale": 0}


# when a whole-file copy was last checked against its source, in an xattr on the file (a replaced file starts over);
# in memory where the filesystem has no user xattrs

XATTR = "user.ycpe.checked"
_checked: dict[str, float] = {}


def checked_at(path: Path) -> float:
  try:
    return float(os.getxattr(path, XATTR))
  except (OSError, ValueError):
    return _checked.get(str(path), 0.0)


def mark_checked(path: Path, when: float | None = None) -> None:
  when = time.time() if when is None else when  # noqa: TID251  outlives the process
  try:
    os.setxattr(path, XATTR, str(when).encode())
  except OSError:
    _checked[str(path)] = when


def fill(key: Key, source: str, url: str, cancelled=None) -> Fill:
  """The download of key in flight, starting one in a thread if there's none. cancelled() can abort a new one."""
  with _fills_lock:
    if (f := _fills.get(key)) is not None:
      return f
    f = _fills[key] = Fill(key, source, url, canonical_url(source, key) or url)
  threading.Thread(target=f.run, args=(cancelled or (lambda: False),), daemon=True, name=f"fill {key}").start()
  return f


def filling(key: Key) -> Fill | None:
  """The download of key in flight, from whichever source."""
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
    with _usage_lock:
      _usage.update(at=time.monotonic(), bytes=total, files=len(files) - removed)
    return {"removed": removed, "freed": freed, "bytes": total}


def prune_soon() -> None:
  """Prunes in a couple of seconds if the running total is over budget (several finishing downloads share one walk)."""
  global _prune_timer
  if totals()[0] > config.cache_budget and (_prune_timer is None or not _prune_timer.is_alive()):
    _prune_timer = threading.Timer(2.0, prune)
    _prune_timer.daemon = True
    _prune_timer.start()


def derived_copies(url: str) -> list[Path]:
  """The other tools' forms of url's file: decompressed logs and url_file chunks (not the whole-file copy)."""
  paths = [Path(cache_file_path(url, comp)) for comp in ("zst", "bz2")]
  if not (c := chunk_paths(url)):
    return paths
  return paths + c[1] + [root() / f"{hash_url(url)}_length"]


def evict(key: Key) -> int:
  """Removes every cached copy of key; returns bytes freed."""
  paths = [device_path(key)]
  for source in ("internal", "api", "openpilotci"):
    url = canonical_url(source, key)
    paths += [Path(cache_file_path(url)), *derived_copies(url)]
  return remove(paths)


def remove(paths: list[Path]) -> int:
  freed = removed = 0
  for p in paths:
    try:
      size = p.stat().st_size
      p.unlink()
    except OSError:
      continue
    freed += size
    removed += 1
  account(-freed, -removed)
  return freed


# endpoints (sources.py resolves; it imports this module, so it's imported lazily here)

module("cache", "1")


def usage() -> dict:
  b, n = totals()
  return {"dir": str(root()), "budget": config.cache_budget, "bytes": b, "files": n,
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
          key = Key(dongle, log_id, seg, name)
          if (hit := lookup(key)):
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
  elif not (isinstance(segs, list) and all(isinstance(n, int) and 0 <= n < 10000 for n in segs)):
    raise HTTPError(400, "bad segs")
  freed = sum(evict(Key(dongle, log_id, n, name)) for n in segs for kind in kinds for name in KINDS[kind])
  return {"freed": freed, **usage()}
