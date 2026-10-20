"""Local openpilot checkouts: cereal schemas and opendbc DBCs to read, DBCs to save back in place, and a change feed.

Checkouts come from --repo (default: the one serve.py runs from) plus, for a git worktree, its main checkout. Each
one's opendbc is <checkout>/opendbc_repo/opendbc/dbc, or the importable opendbc package when that's empty.
"""
import importlib.util
import os
import queue
import re
import subprocess
import threading
import time
from functools import cache
from pathlib import Path

from openpilot.tools.ycpe.daemon.core import HTTPError, config, confine, module, route, sse

DBC_NAME = re.compile(r"^[A-Za-z0-9_.-]+\.dbc\Z")
MAX_DBC = 8 << 20
GIT_TTL = 5.0

module("repos", "1", write=["dbc"])


def git(path: Path, *args: str) -> str:
  try:
    return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, timeout=10).stdout.strip()
  except (OSError, subprocess.TimeoutExpired):
    return ""


_git: dict[str, tuple[float, dict]] = {}


def git_info(path: Path) -> dict | None:
  hit = _git.get(str(path))
  if hit and time.monotonic() - hit[0] < GIT_TTL:
    return hit[1]
  if not git(path, "rev-parse", "--git-dir"):
    return None
  info = {"branch": git(path, "rev-parse", "--abbrev-ref", "HEAD"), "commit": git(path, "rev-parse", "--short=10", "HEAD"),
          "subject": git(path, "log", "-1", "--format=%s"),
          "dirty": bool(git(path, "status", "--porcelain", "--untracked-files=no", "--", "."))}
  _git[str(path)] = (time.monotonic(), info)
  return info


@cache
def package_opendbc() -> Path | None:
  spec = importlib.util.find_spec("opendbc")
  if spec and spec.submodule_search_locations:
    d = Path(list(spec.submodule_search_locations)[0]) / "dbc"
    return d if d.is_dir() else None
  return None


def roots() -> list[Path]:
  out = []
  for r in config.repo_roots:
    p = Path(r).resolve()
    out.append(p)
    common = git(p, "rev-parse", "--path-format=absolute", "--git-common-dir")
    if common and Path(common).name == ".git" and Path(common).parent.resolve() != p:
      out.append(Path(common).parent.resolve())  # a worktree's main checkout
  return list(dict.fromkeys(out))


def checkout(p: Path, i: int) -> dict:
  cereal = next((c for c in (p / "openpilot" / "cereal", p / "cereal") if (c / "log.capnp").is_file()), None)
  dbc = p / "opendbc_repo" / "opendbc" / "dbc"
  if not any(dbc.glob("*.dbc")):
    dbc = package_opendbc()
  return {"id": str(i), "name": p.name, "path": str(p), "git": git_info(p), "cereal": str(cereal) if cereal else None,
          "opendbc": {"dbc": str(dbc), "git": git_info(dbc.parent.parent)} if dbc else None}


def checkouts() -> list[dict]:
  return [checkout(p, i) for i, p in enumerate(roots())]


def get(repo_id: str) -> dict:
  for c in checkouts():
    if c["id"] == repo_id:
      return c
  raise HTTPError(404, "no such checkout")


def dbc_dir(repo_id: str) -> Path:
  c = get(repo_id)
  if not c["opendbc"]:
    raise HTTPError(404, "no opendbc in this checkout")
  return Path(c["opendbc"]["dbc"])


def dbc_path(repo_id: str, name: str) -> Path:
  if not DBC_NAME.match(name):
    raise HTTPError(403, "bad dbc name")
  d = dbc_dir(repo_id)
  return confine(d / name, [str(d)])


@route("GET", "/local/repos")
def list_repos(req):
  return checkouts()


@route("GET", "/local/repos/{id}/capnp")
def capnp(req):
  """The checkout's schema as {import path: text}: every .capnp under cereal (include/c++.capnp too) plus opendbc's
  car/*.capnp, which log.capnp imports as "/car.capnp". Event is the struct log.capnp names Event."""
  c = get(req.params["id"])
  if not c["cereal"]:
    raise HTTPError(404, "no cereal in this checkout")
  root = Path(c["cereal"])
  files = {str(p.relative_to(root)): p.read_text() for p in sorted(root.rglob("*.capnp")) if confine(p, [str(root)])}
  if c["opendbc"]:
    car = Path(c["opendbc"]["dbc"]).parent / "car"
    files.update({p.name: p.read_text() for p in sorted(car.glob("*.capnp")) if p.name not in files})
  return {"repo": c["id"], "cereal": str(root), "git": c["git"], "files": files}


@route("GET", "/local/repos/{id}/dbc")
def dbc_list(req):
  d = dbc_dir(req.params["id"])
  return [{"name": p.name, "size": p.stat().st_size, "mtime": p.stat().st_mtime} for p in sorted(d.glob("*.dbc"))]


@route("GET", "/local/repos/{id}/dbc/{name}")
def dbc_read(req):
  p = dbc_path(req.params["id"], req.params["name"])
  if not p.is_file():
    raise HTTPError(404)
  return {"name": p.name, "text": p.read_text(errors="replace"), "mtime": p.stat().st_mtime}


@route("PUT", "/local/repos/{id}/dbc/{name}")
def dbc_write(req):
  """Saves a DBC into opendbc's dbc directory (a new file or over an existing one). Body: the DBC text. With
  ?mtime=<from the read>, refuses with 409 if the file changed on disk since."""
  p = dbc_path(req.params["id"], req.params["name"])
  if len(req.body) > MAX_DBC:
    raise HTTPError(413)
  try:
    text = req.body.decode()
  except UnicodeDecodeError as e:
    raise HTTPError(400, "not utf-8") from e
  if (want := req.arg("mtime", None, float)) is not None and p.is_file() and abs(p.stat().st_mtime - want) > 1e-3:
    raise HTTPError(409, "changed on disk since it was read")
  tmp = p.with_name(f".{p.name}.ycpe-tmp")
  tmp.write_text(text)
  os.replace(tmp, p)
  return {"name": p.name, "path": str(p), "size": len(req.body), "mtime": p.stat().st_mtime,
          "generated": p.name.endswith("_generated.dbc")}


# change feed: mtimes polled once a second (no inotify dependency)

_watch_subs: set[queue.Queue] = set()
_watch_lock = threading.Lock()
_watcher: threading.Thread | None = None


def _snapshot() -> dict[str, tuple[str, str, float]]:
  out = {}
  for c in checkouts():
    globs = []
    if c["cereal"]:
      globs.append(("capnp", Path(c["cereal"]), "**/*.capnp"))
    if c["opendbc"]:
      globs.append(("dbc", Path(c["opendbc"]["dbc"]), "*.dbc"))
    for kind, d, g in globs:
      for p in d.glob(g):
        try:
          out[str(p)] = (c["id"], kind, p.stat().st_mtime)
        except OSError:
          pass
  return out


def _watch() -> None:
  last = _snapshot()
  while True:
    time.sleep(1.0)
    with _watch_lock:
      if not _watch_subs:
        continue
    now = _snapshot()
    for path in now.keys() | last.keys():
      if now.get(path) != last.get(path):
        repo, kind, _ = now.get(path) or last[path]
        ev = {"repo": repo, "kind": kind, "file": Path(path).name, "path": path, "deleted": path not in now}
        with _watch_lock:
          for q in _watch_subs:
            q.put(ev)
    last = now


@route("GET", "/local/repos/watch")
def watch(req):
  """SSE "change" events {repo, kind: capnp|dbc, file, path, deleted} when cereal or opendbc files change."""
  global _watcher
  with _watch_lock:
    if _watcher is None:
      _watcher = threading.Thread(target=_watch, daemon=True, name="repo watch")
      _watcher.start()
  q: queue.Queue = queue.Queue()

  def events():
    with _watch_lock:
      _watch_subs.add(q)
    try:
      while True:
        try:
          yield "change", q.get(timeout=15)
        except queue.Empty:
          yield None
    finally:
      with _watch_lock:
        _watch_subs.discard(q)
  return sse(events())
