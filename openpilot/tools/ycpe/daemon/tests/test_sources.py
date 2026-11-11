#!/usr/bin/env python3
import http.client
import os
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

# a fake internal data endpoint stands in for every remote source
UPSTREAM = tempfile.mkdtemp()
CACHE = tempfile.mkdtemp()
LOGS = tempfile.mkdtemp()
os.environ["COMMA_CACHE"] = CACHE


class Upstream(SimpleHTTPRequestHandler):
  fail = False  # cut segment 3's rlog, which trickles out, off half way

  def __init__(self, *a, **kw):
    super().__init__(*a, directory=UPSTREAM, **kw)

  def log_message(self, *a):
    pass

  def do_GET(self):
    if self.path.endswith("/3/rlog.zst"):
      data = (Path(UPSTREAM) / self.path.lstrip("/")).read_bytes()
      self.send_response(200)
      self.send_header("Content-Length", str(len(data)))
      self.end_headers()
      for i in range(0, len(data), 100_000):
        if Upstream.fail and i >= len(data) // 2:
          return
        self.wfile.write(data[i:i + 100_000])
        time.sleep(0.02)
      return
    super().do_GET()


up = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
threading.Thread(target=up.serve_forever, daemon=True).start()
os.environ["DATA_ENDPOINT"] = f"http://127.0.0.1:{up.server_address[1]}/"

from openpilot.tools.lib.url_file import URLFile, hash_url
from openpilot.tools.ycpe.daemon import cache
from openpilot.tools.ycpe.daemon.core import config
from openpilot.tools.ycpe.daemon.tests.common import DaemonCase, until
from openpilot.tools.lib import filereader

filereader.DATA_ENDPOINT = os.environ["DATA_ENDPOINT"]  # in case filereader was imported before the env was set

DONGLE, LOG = "0123456789abcdef", "00000001--abcdef0123"
ROUTE = f"{DONGLE}|{LOG}"
ONLY = ["internal"]  # never reach the real API in tests


def blob(seg, name):
  return bytes((seg * 7 + i) % 251 for i in range(300_000 + seg * 1000)) + name.encode()


def forget(seg, name):
  """Drops the cached copies of a file, so the next read downloads it."""
  url = cache.canonical_url("internal", cache.Key(DONGLE, LOG, seg, name))
  for p in Path(CACHE).glob(f"{hash_url(url)}_*"):
    p.unlink()


for seg in range(4):
  for name in ("rlog.zst", "qcamera.ts"):
    p = Path(UPSTREAM) / DONGLE / LOG / str(seg) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(blob(seg, name))
(Path(UPSTREAM) / DONGLE / LOG / "2" / "rlog.zst").unlink()
(Path(UPSTREAM) / DONGLE / LOG / "2" / "qlog.zst").write_bytes(blob(2, "qlog.zst"))


class TestSources(DaemonCase):
  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    os.environ["COMMA_CACHE"] = CACHE
    config.log_roots = [LOGS]

  def test_resolve(self):
    res = self.json("/local/sources/resolve", {"route": ROUTE, "segs": [0, 1, 2, 9], "kinds": ["log", "qcamera"], "sources": ONLY})
    self.assertEqual(res["segments"]["0"]["log"]["file"], "rlog.zst")
    self.assertEqual(res["segments"]["0"]["log"]["size"], len(blob(0, "rlog.zst")))
    self.assertEqual(res["segments"]["2"]["log"]["file"], "qlog.zst")  # mode a falls back per segment
    self.assertEqual(res["missing"]["9"], ["log", "qcamera"])
    self.assertTrue(res["segments"]["0"]["log"]["url"].startswith(f"/local/file/internal/{DONGLE}/{LOG}/0/rlog.zst?tok="))
    res = self.json("/local/sources/resolve", {"route": f"{DONGLE}|00000009--abcdef0123", "kinds": ["rlog"], "sources": ONLY})
    self.assertEqual(res["segments"], {})  # no segments named: the ones on this machine's disk, none for this route

  def test_bad_route(self):
    for route in ("../../etc", f"{ROUTE}/0:3/r", f"{ROUTE}--2"):  # bare routes only: the page names the segments
      self.assertEqual(self.post("/local/sources/resolve", {"route": route})[0].status, 400, route)
    for segs in (["x"], "01", [-1], 3):
      self.json("/local/sources/resolve", {"route": ROUTE, "segs": segs}, status=400)

  def test_file_miss_then_hit(self):
    url = f"/local/file/internal/{DONGLE}/{LOG}/1/rlog.zst"
    r, data = self.req(url)
    self.assertEqual((r.status, data), (200, blob(1, "rlog.zst")))
    canonical = cache.canonical_url("internal", cache.Key(DONGLE, LOG, 1, "rlog.zst"))
    until(lambda: cache.lookup(cache.Key(DONGLE, LOG, 1, "rlog.zst")), 1)
    r, data = self.req(url, headers={"Range": "bytes=5-9"})
    self.assertEqual((r.status, data, r.getheader("X-Ycpe-Cache")), (206, blob(1, "rlog.zst")[5:10], "hit"))
    # the python tools read the same download from their chunk cache, without the network
    self.assertEqual(URLFile(canonical + "?sig=x").read(), blob(1, "rlog.zst"))

  def test_video_in_url_file_chunks(self):
    r, data = self.req(f"/local/file/internal/{DONGLE}/{LOG}/0/qcamera.ts")
    self.assertEqual((r.status, data), (200, blob(0, "qcamera.ts")))
    until(lambda: cache.lookup(cache.Key(DONGLE, LOG, 0, "qcamera.ts")), 1)
    canonical = cache.canonical_url("internal", cache.Key(DONGLE, LOG, 0, "qcamera.ts"))
    self.assertEqual(URLFile(canonical).read(), blob(0, "qcamera.ts"))
    self.assertIn(f"{hash_url(canonical)}_0.0", (Path(CACHE) / "manifest.txt").read_text())  # under url_file's LRU

  def test_concurrent_readers_during_fill(self):
    url = f"/local/file/internal/{DONGLE}/{LOG}/3/rlog.zst"
    with ThreadPoolExecutor(2) as ex:
      out = list(ex.map(lambda _: self.req(url)[1], range(2)))
    self.assertEqual(out, [blob(3, "rlog.zst")] * 2)
    forget(3, "rlog.zst")

  @mock.patch.object(Upstream, "fail", True)
  def test_failed_fill_is_an_error(self):
    forget(3, "rlog.zst")
    with self.assertRaises(http.client.IncompleteRead):
      self.req(f"/local/file/internal/{DONGLE}/{LOG}/3/rlog.zst")
    self.assertIsNone(cache.lookup(cache.Key(DONGLE, LOG, 3, "rlog.zst")))

  def test_file_rejects(self):
    for path in (f"/local/file/internal/{DONGLE}/{LOG}/0/..%2f..%2fx", f"/local/file/internal/{DONGLE}/{LOG}/0/passwd",
                 f"/local/file/nope/{DONGLE}/{LOG}/0/rlog.zst", f"/local/file/internal/zz/{LOG}/0/rlog.zst",
                 f"/local/file/local/{DONGLE}/..%2f..%2f/0/rlog.zst"):
      self.assertIn(self.status(path), (403, 404), path)

  def test_local_source(self):
    d = Path(LOGS) / f"{LOG}--0"
    d.mkdir(exist_ok=True)
    (d / "qlog.zst").write_bytes(b"local qlog")
    res = self.json("/local/sources/resolve", {"route": ROUTE, "segs": [0], "kinds": ["qlog"], "sources": ["local"]})
    self.assertEqual(res["segments"]["0"]["qlog"]["source"], "local")
    self.assertEqual(self.req(f"/local/file/local/{DONGLE}/{LOG}/0/qlog.zst")[1], b"local qlog")

  def test_local_symlink_confined(self):
    outside = Path(tempfile.mkdtemp())
    (outside / "qlog.zst").write_bytes(b"outside")
    link = Path(LOGS) / f"{LOG}--5"
    if not link.exists():
      link.symlink_to(outside)
    self.assertEqual(self.status(f"/local/file/local/{DONGLE}/{LOG}/5/qlog.zst"), 403)


if __name__ == "__main__":
  unittest.main()
