"""Local openpilot checkouts: cereal schemas and opendbc DBCs to read, DBCs to save back in place, and a change feed.

Checkouts come from --repo (default: the one serve.py runs from), by their index as id. Each one's DBCs are its own
opendbc_repo/opendbc/dbc and nothing else, so a save never lands outside the checkout picked.
"""
import os
import re
import subprocess
import threading
import time
from pathlib import Path

from openpilot.tools.ycpe.daemon.core import Feed, HTTPError, config, confine, module, route

DBC_NAME = re.compile(r"^[A-Za-z0-9_.-]+\.dbc\Z")
MAX_DBC = 8 << 20
OPENDBC = Path("opendbc_repo") / "opendbc"

module("repos", "1")


def roots() -> list[Path]:
  return list(dict.fromkeys(Path(r).resolve() for r in config.repo_roots))


def root(repo_id: str) -> Path:
  for i, p in enumerate(roots()):
    if str(i) == repo_id:
      return p
  raise HTTPError(404, "no such checkout")


def cereal_dir(p: Path) -> Path | None:
  return next((c for c in (p / "openpilot" / "cereal", p / "cereal") if (c / "log.capnp").is_file()), None)


def dbc_dir(p: Path) -> Path | None:
  d = p / OPENDBC / "dbc"
  return d if any(d.glob("*.dbc")) else None  # none in an empty submodule


def git_info(p: Path) -> dict | None:
  """{branch, commit, dirty} from one git call, live: the checkout may be switched or edited at any time"""
  try:
    out = subprocess.run(["git", "-C", str(p), "status", "--porcelain=v2", "--branch", "--untracked-files=no"],
                         capture_output=True, text=True, timeout=10)
  except (OSError, subprocess.TimeoutExpired):
    return None
  if out.returncode:
    return None
  lines = out.stdout.splitlines()
  head = dict(line[2:].split(" ", 1) for line in lines if line.startswith("# "))
  return {"branch": head.get("branch.head"), "commit": head.get("branch.oid", "")[:10], "dirty": any(not line.startswith("#") for line in lines)}


def dbcs(repo_id: str) -> Path:
  if (d := dbc_dir(root(repo_id))) is None:
    raise HTTPError(404, "no opendbc in this checkout")
  return d


def dbc_path(repo_id: str, name: str) -> Path:
  if not DBC_NAME.match(name):
    raise HTTPError(403, "bad dbc name")
  d = dbcs(repo_id)
  return confine(d / name, [str(d)])


@route("GET", "/local/repos")
def list_repos(req):
  out = []
  for i, p in enumerate(roots()):
    cereal, dbc = cereal_dir(p), dbc_dir(p)
    out.append({"id": str(i), "name": p.name, "git": git_info(p), "cereal": str(cereal) if cereal else None,
                "opendbc": {"dbc": str(dbc)} if dbc else None})
  return out


@route("GET", "/local/repos/{id}/capnp")
def capnp(req):
  """The checkout's schema as {files: {import path: text}}: every .capnp under cereal (include/c++.capnp too) plus
  opendbc's car/*.capnp, which log.capnp imports as "/car.capnp"."""
  p = root(req.params["id"])
  if (cereal := cereal_dir(p)) is None:
    raise HTTPError(404, "no cereal in this checkout")
  files = {str(f.relative_to(cereal)): f.read_text() for f in sorted(cereal.rglob("*.capnp")) if confine(f, [str(cereal)])}
  files.update({f.name: f.read_text() for f in sorted((p / OPENDBC / "car").glob("*.capnp")) if f.name not in files})
  return {"files": files}


@route("GET", "/local/repos/{id}/dbc")
def dbc_list(req):
  return [{"name": p.name} for p in sorted(dbcs(req.params["id"]).glob("*.dbc"))]


@route("GET", "/local/repos/{id}/dbc/{name}")
def dbc_read(req):
  p = dbc_path(req.params["id"], req.params["name"])
  if not p.is_file():
    raise HTTPError(404)
  return {"name": p.name, "text": p.read_text(errors="replace"), "mtime": p.stat().st_mtime}


@route("PUT", "/local/repos/{id}/dbc/{name}", max_body=MAX_DBC)
def dbc_write(req):
  """Saves a DBC into opendbc's dbc directory (a new file or over an existing one). Body: the DBC text. With
  ?mtime=<from the read>, refuses with 409 if the file changed on disk since."""
  p = dbc_path(req.params["id"], req.params["name"])
  try:
    text = req.body.decode()
  except UnicodeDecodeError as e:
    raise HTTPError(400, "not utf-8") from e
  if (want := req.arg("mtime", None, float)) is not None and p.is_file() and abs(p.stat().st_mtime - want) > 1e-3:
    raise HTTPError(409, "changed on disk since it was read")
  tmp = p.with_name(f".{p.name}.ycpe-tmp")
  tmp.write_text(text)
  os.replace(tmp, p)
  return {"name": p.name, "path": str(p), "mtime": p.stat().st_mtime, "generated": p.name.endswith("_generated.dbc")}


# change feed: cereal's .capnp mtimes polled once a second (no inotify dependency), with no git calls

_changes = Feed()
_watch_lock = threading.Lock()
_watcher: threading.Thread | None = None


def _snapshot() -> dict[str, tuple[str, float]]:
  out = {}
  for i, p in enumerate(roots()):
    if (cereal := cereal_dir(p)) is None:
      continue
    for f in cereal.glob("**/*.capnp"):
      try:
        out[str(f)] = (str(i), f.stat().st_mtime)
      except OSError:
        pass
  return out


def _watch() -> None:
  last = _snapshot()
  while True:
    time.sleep(1.0)
    if not _changes.subs:
      continue
    now = _snapshot()
    for path in now.keys() | last.keys():
      if now.get(path) != last.get(path):
        repo, _ = now.get(path) or last[path]
        ev = {"repo": repo, "kind": "capnp", "file": Path(path).name, "path": path, "deleted": path not in now}
        _changes.publish(ev)
    last = now


@route("GET", "/local/repos/watch")
def watch(req):
  """SSE "change" events {repo, kind: capnp, file, path, deleted} when cereal's schema files change."""
  global _watcher
  with _watch_lock:
    if _watcher is None:
      _watcher = threading.Thread(target=_watch, daemon=True, name="repo watch")
      _watcher.start()
  return _changes.stream("change")
