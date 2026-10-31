"""Local openpilot checkouts: their opendbc DBCs to read, and to save back in place.

Checkouts come from --repo (default: the one serve.py runs from), by their index as id. Each one's DBCs are its own
opendbc_repo/opendbc/dbc and nothing else, so a save never lands outside the checkout picked.
"""
import os
import re
from pathlib import Path

from openpilot.tools.ycpe.daemon.core import HTTPError, config, confine, route

DBC_NAME = re.compile(r"^[A-Za-z0-9_.-]+\.dbc\Z")
MAX_DBC = 8 << 20


def roots() -> list[Path]:
  return list(dict.fromkeys(Path(r).resolve() for r in config.repo_roots))


def root(repo_id: str) -> Path:
  for i, p in enumerate(roots()):
    if str(i) == repo_id:
      return p
  raise HTTPError(404, "no such checkout")


def dbc_dir(p: Path) -> Path | None:
  d = p / "opendbc_repo" / "opendbc" / "dbc"
  return d if any(d.glob("*.dbc")) else None  # none in an empty submodule


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
  return [{"id": str(i), "name": p.name, "opendbc": {"dbc": str(d)} if (d := dbc_dir(p)) else None} for i, p in enumerate(roots())]


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

