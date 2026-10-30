#!/usr/bin/env python3
"""Serves the app on 127.0.0.1, plus the optional local daemon (/local/*: auth, sources, shared cache, repos,
transcode). With --static, or without openpilot's Python deps, it's a plain static server and the app degrades gracefully."""
import argparse
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2]))  # the directory holding the openpilot package


class StaticHandler(SimpleHTTPRequestHandler):
  def do_GET(self):
    path = self.path.split("?")[0]
    if not Path(path).suffix and not (ROOT / path.lstrip("/")).is_file():
      self.path = "/index.html"
    super().do_GET()

  def end_headers(self):
    self.send_header("Cache-Control", "no-store")
    super().end_headers()


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("port", nargs="?", type=int, default=3000, help="athena allows localhost:3000; elsewhere the daemon relays its calls")
  p.add_argument("--static", action="store_true", help="no daemon, static files only")
  p.add_argument("--host", default="127.0.0.1", help="this PC's LAN address, to share it on the network (the daemon too: its files, cache and sign-in)")
  p.add_argument("--cache-budget", type=float, default=20, help="GB the shared download cache may use (LRU pruned)")
  p.add_argument("--data-dir", action="append", default=[], help="local route directory the daemon may read (repeatable)")
  p.add_argument("--repo", action="append", default=[], help="openpilot checkout (default: this one)")
  p.add_argument("--webrtcd-port", type=int, default=5001, help="port of this PC's webrtcd, for live data at /live/local")
  args = p.parse_args()

  if not args.static:
    try:
      from openpilot.common.hardware.hw import Paths
      from openpilot.tools.ycpe.daemon.core import session_token
      from openpilot.tools.ycpe.daemon.server import make_server, config
    except ImportError as e:
      print(f"daemon unavailable ({e}); serving static files only. Run with openpilot's venv for the daemon.")
      args.static = True

  if args.static:
    server = ThreadingHTTPServer((args.host, args.port), partial(StaticHandler, directory=str(ROOT)))
  else:
    config.token = session_token()
    config.cache_budget = int(args.cache_budget * 1000 ** 3)
    config.log_roots = [Paths.log_root(), *args.data_dir]
    config.repo_roots = args.repo or [str(ROOT.parents[2])]
    config.webrtcd_port = args.webrtcd_port
    config.host = args.host if args.host != "127.0.0.1" else None
    server = make_server(args.port)
  print(f"http://{'localhost' if args.host == '127.0.0.1' else args.host}:{args.port}/" + ("" if args.static else "  (with the local daemon)"))
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    pass


if __name__ == "__main__":
  main()
