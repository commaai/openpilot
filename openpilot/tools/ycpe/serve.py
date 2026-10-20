#!/usr/bin/env python3
"""Dev server: static files, with unknown routes (paths without a file extension) falling back to index.html."""
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class Handler(SimpleHTTPRequestHandler):
  def do_GET(self):
    path = self.path.split("?")[0]
    if not Path(path).suffix and not (ROOT / path.lstrip("/")).is_file():
      self.path = "/index.html"
    super().do_GET()

  def end_headers(self):
    self.send_header("Cache-Control", "no-store")
    super().end_headers()


if __name__ == "__main__":
  port = int(sys.argv[1]) if len(sys.argv) > 1 else 3000  # athena only allows localhost:3000
  print(f"http://localhost:{port}/")
  ThreadingHTTPServer(("", port), partial(Handler, directory=str(ROOT))).serve_forever()
