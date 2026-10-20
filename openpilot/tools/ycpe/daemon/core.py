"""The daemon's HTTP core: config, route registry, request/response types, security checks and streaming."""
import json
import mimetypes
import os
import re
import secrets
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from pathlib import Path
from urllib.parse import parse_qsl, unquote

TOKEN_HEADER = "X-Ycpe-Token"
TOKEN_PARAM = "tok"  # for URLs the page can't set headers on (<video src>, a worker's fetch of a resolved url)
CHUNK = 1 << 20
TOKEN_FILE = Path.home() / ".cache" / "ycpe" / "token"


@dataclass
class Config:
  port: int = 3000
  token: str = field(default_factory=lambda: secrets.token_urlsafe(32))
  cache_root: str = ""  # the shared download cache; set from Paths.download_cache_root() by serve.py
  cache_budget: int = 20 * 1000 ** 3
  log_roots: list[str] = field(default_factory=list)  # local route directories that may be read
  repo_roots: list[str] = field(default_factory=list)  # openpilot checkouts
  device: str | None = None  # [user@]host of a comma device for /local/device
  device_key: str | None = None
  device_root: str | None = None  # the device's realdata directory if not the usual one (tests use a stand-in)

  @property
  def origins(self) -> set[str]:
    return {f"http://{h}:{self.port}" for h in ("localhost", "127.0.0.1")}

  @property
  def hosts(self) -> set[str]:
    return {f"{h}:{self.port}" for h in ("localhost", "127.0.0.1")}


config = Config()


def session_token(path: Path = TOKEN_FILE) -> str:
  """The session token kept across restarts (only this user can read it), so open tabs keep working. A file others
  can read, or write, gets a new one."""
  try:
    if not path.stat().st_mode & 0o077 and len(token := path.read_text().strip()) >= 32:
      return token
  except OSError:
    pass
  token = secrets.token_urlsafe(32)
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_name(f".token.{os.getpid()}")
  with open(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
    f.write(token)
  os.replace(tmp, path)
  return token


class HTTPError(Exception):
  def __init__(self, status: int, message: str = ""):
    super().__init__(message or HTTPStatus(status).phrase)
    self.status = status


class NoRoute(HTTPError):
  """No /local/ route matches the path (as opposed to a handler's own 404)."""


@dataclass
class Request:
  method: str
  path: str
  params: dict[str, str]  # from {name} in the route pattern
  query: dict[str, str]
  headers: dict[str, str]  # lower-case names
  body: bytes
  handler: object = None  # the BaseHTTPRequestHandler, for raw access

  def json(self):
    try:
      return json.loads(self.body or b"null")
    except ValueError as e:
      raise HTTPError(400, "bad json") from e

  def arg(self, name: str, default=None, kind=str):
    v = self.query.get(name, default)
    try:
      return v if v is None or v is default else kind(v)
    except ValueError as e:
      raise HTTPError(400, f"bad {name}") from e


# handlers return one of these, or a dict/list (sent as JSON), or None (204)
@dataclass
class Response:
  body: bytes = b""
  status: int = 200
  type: str = "application/octet-stream"
  headers: dict[str, str] = field(default_factory=dict)


@dataclass
class Stream:
  """A streamed body: chunked, or with Content-Length when length is known. chunks yields bytes and is closed when
  the client goes away. If it raises, the connection is cut short so the client sees an error, not a short body."""
  chunks: object
  type: str = "application/octet-stream"
  headers: dict[str, str] = field(default_factory=dict)
  status: int = 200
  length: int | None = None


@dataclass
class File:
  """A file on disk with Range support. While growing() is true the file is still being written: the response is
  chunked (or sized, if the final size is known) and follows the file's tail until growing() turns false. growing()
  may raise to abort the response, e.g. when the download feeding the file fails."""
  path: str
  type: str | None = None
  headers: dict[str, str] = field(default_factory=dict)
  growing: object = None  # callable -> bool
  size: int | None = None  # the final size of a growing file, when known


def sse(events, keepalive: float = 15.0):
  """Server-sent events from an iterator of (event, data) pairs; data is JSON-encoded. Yield None to send a
  keepalive comment. The iterator should block between events (e.g. on a queue with a timeout)."""
  def gen():
    yield b"retry: 2000\n\n"
    for item in events:
      if item is None:
        yield b": keepalive\n\n"
        continue
      event, data = item
      yield f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode()
  return Stream(gen(), "text/event-stream", {"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


# routes

_routes: list[tuple[str, re.Pattern, object, bool, int]] = []
MAX_BODY = 64 << 20
modules: dict[str, dict] = {}  # name -> {version, ...info}, listed by /local/caps


def _compile(pattern: str) -> re.Pattern:
  # {name} matches one path segment, {name:path} the rest of the path
  def sub(m):
    name, kind = m.group(1), m.group(2)
    return f"(?P<{name}>.+)" if kind == "path" else f"(?P<{name}>[^/]+)"
  return re.compile("^" + re.sub(r"\{(\w+)(?::(\w+))?\}", sub, pattern) + r"\Z")


def route(method: str, pattern: str, public: bool = False, max_body: int = MAX_BODY):
  """Registers fn(req) for method and pattern under /local/. public routes skip the session token (only caps); a
  request body over max_body bytes is refused before it's read."""
  assert pattern.startswith("/local/"), pattern
  def deco(fn):
    _routes.append((method, _compile(pattern), fn, public, max_body))
    return fn
  return deco


def module(name: str, version: str = "1", **info):
  """Declares an enabled feature module; the page lights features up from /local/caps."""
  modules[name] = {"version": version, **info}


def match(method: str, path: str):
  allowed = False
  for m, rx, fn, public, max_body in _routes:
    if (g := rx.match(path)):
      if m == method:
        return fn, {k: unquote(v) for k, v in g.groupdict().items()}, public, max_body
      allowed = True
  raise HTTPError(405) if allowed else NoRoute(404)


# security

def check_host(headers: dict[str, str]) -> None:
  # DNS rebinding: a page on evil.example resolved to 127.0.0.1 still sends its own name as Host
  if headers.get("host", "").lower() not in config.hosts:
    raise HTTPError(403, "bad host")


def check_local(headers: dict[str, str], query: dict[str, str], public: bool) -> None:
  origin = headers.get("origin")
  if origin is not None and origin not in config.origins:
    raise HTTPError(403, "bad origin")
  if headers.get("sec-fetch-site") not in (None, "same-origin", "none"):
    raise HTTPError(403, "cross-site")
  if public:
    return
  token = headers.get(TOKEN_HEADER.lower()) or query.get(TOKEN_PARAM) or ""
  if not secrets.compare_digest(token.encode(), config.token.encode()):
    raise HTTPError(403, "missing or wrong token")


def confine(path: str | os.PathLike, roots: list[str]) -> Path:
  """The real path of path if it lies inside one of roots (after resolving symlinks and ..), else 403."""
  p = str(path)
  if "\0" in p or any(part == ".." for part in re.split(r"[\\/]", p)):
    raise HTTPError(403, "path traversal")
  real = os.path.realpath(p)
  for root in roots:
    r = os.path.realpath(root)
    if real == r or real.startswith(r.rstrip(os.sep) + os.sep):
      return Path(real)
  raise HTTPError(403, "outside allowed directories")


# responses

def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
  """[start, end) for a single bytes range, None for the whole file; 416 if unsatisfiable. Multiple ranges are
  answered with the whole file, which RFC 9110 allows."""
  if not header or not header.startswith("bytes=") or "," in header:
    return None
  a, _, b = header[6:].strip().partition("-")
  try:
    if a == "":
      n = int(b)
      start, end = max(0, size - n), size
    else:
      start, end = int(a), (min(size, int(b) + 1) if b else size)
  except ValueError:
    return None
  if start >= size or start >= end:
    raise HTTPError(416)
  return start, end


def guess_type(path: str) -> str:
  if path.endswith((".zst", ".bz2", ".hevc", ".ts")):
    return "application/octet-stream"
  return mimetypes.guess_type(path)[0] or "application/octet-stream"


def file_chunks(src, start: int, end: int | None, growing=None, poll: float = 0.05):
  """Bytes [start, end) of src (a path, or a file opened in binary mode, which is closed at the end). With growing,
  waits for a file being written until growing() turns false."""
  path = getattr(src, "name", src)
  with (open(src, "rb") if isinstance(src, (str, os.PathLike)) else src) as f:
    f.seek(start)
    pos = start
    while end is None or pos < end:
      data = f.read(CHUNK if end is None else min(CHUNK, end - pos))
      if data:
        pos += len(data)
        yield data
      elif growing is not None and growing():
        time.sleep(poll)
      elif growing is not None:
        growing = None  # finished: one more pass picks up what was written since the last read
      elif end is not None:
        raise OSError(f"{path} ended at {pos} of {end}")
      else:
        return


def parse_query(qs: str) -> dict[str, str]:
  return dict(parse_qsl(qs, keep_blank_values=True))
