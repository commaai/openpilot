"""The HTTP server: the static app (with the session token in index.html) plus the /local/* API."""
import html
import importlib
import json
import os
import subprocess
import sys
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from openpilot.tools.ycpe.daemon import core
from openpilot.tools.ycpe.daemon.core import HTTPError, Response, Stream, File, config

APP = Path(__file__).resolve().parent.parent
# feature modules, imported in order; one that fails to import is left out of /local/caps with its error
MODULES = ["auth", "jobs", "athena", "billing", "webrtc", "sources", "cache", "repos", "device", "transcode", "replay"]
failed: dict[str, str] = {}


def load_modules() -> None:
  for name in MODULES:
    try:
      importlib.import_module(f"openpilot.tools.ycpe.daemon.{name}")
    except ModuleNotFoundError as e:
      if e.name != f"openpilot.tools.ycpe.daemon.{name}":
        failed[name] = f"{type(e).__name__}: {e}"
    except Exception as e:
      failed[name] = f"{type(e).__name__}: {e}"
      traceback.print_exc()


def app_commit() -> dict | None:
  """the commit the app is served from, and its time: the account menu's version"""
  try:
    out = subprocess.run(["git", "-C", str(APP), "log", "-1", "--format=%H %ct"], capture_output=True, text=True, timeout=5).stdout.split()
  except (OSError, subprocess.TimeoutExpired):
    return None
  return {"sha": out[0], "time": int(out[1])} if len(out) == 2 else None


@core.route("GET", "/local/caps", public=True)
def caps(req):
  return {"daemon": "ycpe", "version": 1, "modules": core.modules, "failed": failed, "commit": app_commit()}


class Handler(SimpleHTTPRequestHandler):
  protocol_version = "HTTP/1.1"
  disable_nagle_algorithm = True  # headers and a sendfile body are separate writes: without this, 40 ms delayed ACKs
  peer_checked = False  # one handler per connection: its owner is looked up once

  def __init__(self, *args, **kwargs):
    super().__init__(*args, directory=str(APP), **kwargs)

  def log_message(self, fmt, *args):
    # never log query strings: resolved urls carry the session token
    sys.stderr.write(f"{self.address_string()} {self.command} {self.path.split('?')[0]} {args[1] if len(args) > 1 else ''}\n")

  def end_headers(self):
    self.send_header("Cache-Control", "no-store")
    self.send_header("Referrer-Policy", "no-referrer")
    self.send_header("X-Content-Type-Options", "nosniff")
    self.send_header("X-Frame-Options", "DENY")  # the page runs signed in: no other site may frame it
    self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
    super().end_headers()

  def _headers(self) -> dict[str, str]:
    return {k.lower(): v for k, v in self.headers.items()}

  def do_GET(self):
    self.dispatch("GET")

  def do_HEAD(self):
    self.dispatch("HEAD")

  def do_POST(self):
    self.dispatch("POST")

  def do_PUT(self):
    self.dispatch("PUT")

  def do_DELETE(self):
    self.dispatch("DELETE")

  def dispatch(self, method: str):
    path, _, qs = self.path.partition("?")
    headers = self._headers()
    try:
      if not self.peer_checked:
        core.check_peer(self.client_address, self.connection.getsockname())
        self.peer_checked = True
      core.check_host(headers)  # every request, static too: index.html carries the token
      if path.startswith("/local/"):
        try:
          return self.api(method, path, core.parse_query(qs), headers)
        except core.NoRoute:
          # the app has page routes under /local/ too (loose files): a reload of one gets the app, not an API 404
          navigating = headers.get("sec-fetch-mode") == "navigate" or "text/html" in headers.get("accept", "")
          if method in ("GET", "HEAD") and navigating:
            return self.index(method)
          raise
      if method not in ("GET", "HEAD"):
        raise HTTPError(405)
      if (not Path(path).suffix and not (APP / path.lstrip("/")).is_file()) or path == "/index.html":
        return self.index(method)
      if "/." in path:
        raise HTTPError(404)
      super().do_GET() if method == "GET" else super().do_HEAD()
    except HTTPError as e:
      self.error(e.status, str(e))
    except (BrokenPipeError, ConnectionResetError):
      pass

  def index(self, method: str):
    page = (APP / "index.html").read_text()
    meta = f'<meta name="ycpe-daemon" content="{html.escape(config.token)}">'
    body = page.replace("<head>", "<head>\n  " + meta, 1).encode()
    self.reply(Response(body, type="text/html; charset=utf-8"), head=method == "HEAD")

  def api(self, method: str, path: str, query: dict, headers: dict):
    fn, params, public, max_body = core.match("GET" if method == "HEAD" else method, path)
    core.check_local(headers, query, public)
    n = int(headers.get("content-length") or 0)
    if n > max_body:
      raise HTTPError(413)
    body = self.rfile.read(n) if n else b""
    req = core.Request(method, path, params, query, headers, body, self)
    try:
      out = fn(req)
    except HTTPError:
      raise
    except Exception as e:
      traceback.print_exc()
      raise HTTPError(500, f"{type(e).__name__}: {e}") from e
    if out is None:
      out = Response(status=204)
    elif isinstance(out, (dict, list)):
      out = Response(json.dumps(out).encode(), type="application/json")
    if isinstance(out, Response):
      self.reply(out, head=method == "HEAD")
    elif isinstance(out, Stream):
      self.stream(out.chunks, out.status, out.type, out.headers, head=method == "HEAD", length=out.length)
    elif isinstance(out, File):
      self.file(out, headers.get("range"), head=method == "HEAD")
    else:
      raise HTTPError(500, f"bad handler result {type(out).__name__}")

  def reply(self, r: Response, head=False):
    self.send_response(r.status)
    self.send_header("Content-Type", r.type)
    for k, v in r.headers.items():
      self.send_header(k, v)
    self.send_header("Content-Length", str(len(r.body)))
    self.end_headers()
    if not head and r.status != 204:
      self.wfile.write(r.body)

  def error(self, status: int, message: str):
    self.close_connection = True
    try:
      self.reply(Response(json.dumps({"error": message}).encode(), status, "application/json"))
    except (BrokenPipeError, ConnectionResetError):
      pass

  def stream(self, chunks, status, ctype, extra, head=False, length=None):
    self.send_response(status)
    self.send_header("Content-Type", ctype)
    for k, v in extra.items():
      self.send_header(k, v)
    self.send_header(*(("Content-Length", str(length)) if length is not None else ("Transfer-Encoding", "chunked")))
    self.end_headers()
    try:
      if head:
        return
      for data in chunks:
        if data:
          self.wfile.write(data if length is not None else b"%x\r\n%b\r\n" % (len(data), data))
          self.wfile.flush()
      if length is None:
        self.wfile.write(b"0\r\n\r\n")
    except (BrokenPipeError, ConnectionResetError):
      self.close_connection = True
    except Exception as e:
      sys.stderr.write(f"stream aborted: {type(e).__name__}: {e}\n")
      self.close_connection = True  # no terminating chunk: the client sees a failed body
    finally:
      if hasattr(chunks, "close"):
        chunks.close()

  def file(self, f: File, range_header: str | None, head=False):
    if not os.path.isfile(f.path):
      raise HTTPError(404)
    ctype = f.type or core.guess_type(f.path)
    if f.growing is not None and f.growing():
      return self.stream(core.file_chunks(f.path, 0, f.size, f.growing), 200, ctype, f.headers, head, f.size)
    size = os.path.getsize(f.path)
    rng = core.parse_range(range_header, size)
    start, end = rng or (0, size)
    self.send_response(206 if rng else 200)
    self.send_header("Content-Type", ctype)
    self.send_header("Accept-Ranges", "bytes")
    if rng:
      self.send_header("Content-Range", f"bytes {start}-{end - 1}/{size}")
    for k, v in f.headers.items():
      self.send_header(k, v)
    self.send_header("Content-Length", str(end - start))
    self.end_headers()
    if head:
      return
    try:
      with open(f.path, "rb") as fh:
        self.connection.sendfile(fh, start, end - start)
    except OSError:
      self.close_connection = True


class Server(ThreadingHTTPServer):
  daemon_threads = True
  allow_reuse_address = True
  request_queue_size = 64


def make_server(port: int) -> Server:
  config.port = port
  load_modules()
  return Server(("0.0.0.0" if config.host else "127.0.0.1", port), Handler)  # loopback only, unless shared with --host
