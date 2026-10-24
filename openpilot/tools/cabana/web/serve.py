#!/usr/bin/env python3
"""Serves dist/ with the cross-origin isolation headers that SharedArrayBuffer (pthreads) needs."""
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Handler(SimpleHTTPRequestHandler):
  def end_headers(self):
    self.send_header("Cross-Origin-Opener-Policy", "same-origin")
    self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
    self.send_header("Cache-Control", "no-store")
    super().end_headers()


if __name__ == "__main__":
  port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
  Handler.extensions_map[".wasm"] = "application/wasm"
  handler = partial(Handler, directory=str(Path(__file__).resolve().parent / "dist"))
  print(f"http://localhost:{port}/cabana.html")
  ThreadingHTTPServer(("", port), handler).serve_forever()
