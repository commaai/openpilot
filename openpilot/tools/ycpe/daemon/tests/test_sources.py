#!/usr/bin/env python3
import http.client
import json
import os
import tempfile
import threading
import time
import unittest
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# a fake internal data endpoint stands in for every remote source
UPSTREAM = tempfile.mkdtemp()
CACHE = tempfile.mkdtemp()
LOGS = tempfile.mkdtemp()
os.environ["COMMA_CACHE"] = CACHE


class Upstream(SimpleHTTPRequestHandler):
  slow = threading.Event()  # set: the next "slow" file trickles out
  fail = False  # cut the slow file off half way

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
    if self.path.startswith("/redirect"):
      self.send_response(302)
      self.send_header("Location", "https://evil.example/x")
      self.send_header("Content-Length", "0")
      self.end_headers()
      return
    super().do_GET()


up = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
threading.Thread(target=up.serve_forever, daemon=True).start()
os.environ["DATA_ENDPOINT"] = f"http://127.0.0.1:{up.server_address[1]}/"

from openpilot.tools.lib.url_file import URLFile, hash_url
from openpilot.tools.ycpe.daemon import cache, jobs
from openpilot.tools.ycpe.daemon.core import config
from openpilot.tools.ycpe.daemon.tests.test_security import DaemonCase
from openpilot.tools.lib import filereader

filereader.DATA_ENDPOINT = os.environ["DATA_ENDPOINT"]  # in case filereader was imported before the env was set

DONGLE, LOG = "0123456789abcdef", "00000001--abcdef0123"
ROUTE = f"{DONGLE}|{LOG}"
ONLY = ["internal"]  # never reach the real API in tests


def blob(seg, name):
  return bytes((seg * 7 + i) % 251 for i in range(300_000 + seg * 1000)) + name.encode()


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
    config.cache_root = CACHE + "/"
    config.log_roots = [LOGS]

  def json(self, path, body=None):
    r, data = self.req(path, method="POST" if body is not None else "GET", body=json.dumps(body) if body is not None else None)
    self.assertEqual(r.status, 200, data)
    return json.loads(data)

  def test_resolve(self):
    res = self.json("/local/sources/resolve", {"route": ROUTE, "segs": [0, 1, 2, 9], "kinds": ["log", "qcamera"], "sources": ONLY})
    self.assertEqual(res["segments"]["0"]["log"]["file"], "rlog.zst")
    self.assertEqual(res["segments"]["0"]["log"]["size"], len(blob(0, "rlog.zst")))
    self.assertEqual(res["segments"]["2"]["log"]["file"], "qlog.zst")  # mode a falls back per segment
    self.assertEqual(res["missing"]["9"], ["log", "qcamera"])
    res = self.json("/local/sources/resolve", {"route": f"{ROUTE}/0:3/r", "kinds": ["log"], "sources": ONLY, "segs": [0, 2]})
    self.assertEqual(res["missing"], {"2": ["log"]})  # /r: no qlog fallback
    self.assertTrue(res["segments"]["0"]["log"]["url"].startswith(f"/local/file/internal/{DONGLE}/{LOG}/0/rlog.zst?tok="))

  def test_bad_route(self):
    r, _ = self.req("/local/sources/resolve", method="POST", body=json.dumps({"route": "../../etc"}))
    self.assertEqual(r.status, 400)

  def test_file_miss_then_hit(self):
    url = f"/local/file/internal/{DONGLE}/{LOG}/1/rlog.zst"
    r, data = self.req(url)
    self.assertEqual((r.status, data), (200, blob(1, "rlog.zst")))
    canonical = cache.canonical_url("internal", cache.Key(DONGLE, LOG, 1, "rlog.zst"))
    for _ in range(50):
      if (Path(CACHE) / f"{hash_url(canonical)}_length").exists():
        break
      time.sleep(0.02)
    self.assertTrue((Path(CACHE) / cache.sha(canonical)).is_file())
    hits = cache.stats["hits"]
    r, data = self.req(url, headers={"Range": "bytes=5-9"})
    self.assertEqual((r.status, data, r.getheader("X-Ycpe-Cache")), (206, blob(1, "rlog.zst")[5:10], "hit"))
    self.assertEqual(cache.stats["hits"], hits + 1)
    # the python tools read the same download from their chunk cache, without the network
    self.assertEqual(URLFile(canonical + "?sig=x").read(), blob(1, "rlog.zst"))

  def test_concurrent_readers_during_fill(self):
    Upstream.fail = False
    url = f"/local/file/internal/{DONGLE}/{LOG}/3/rlog.zst"
    out = [None, None]

    def get(i):
      out[i] = self.req(url)[1]
    ts = [threading.Thread(target=get, args=(i,)) for i in range(2)]
    for t in ts:
      t.start()
    for t in ts:
      t.join()
    self.assertEqual(out, [blob(3, "rlog.zst")] * 2)
    cache.evict(cache.Key(DONGLE, LOG, 3, "rlog.zst"))

  def test_failed_fill_is_an_error(self):
    Upstream.fail = True
    try:
      cache.evict(cache.Key(DONGLE, LOG, 3, "rlog.zst"))
      with self.assertRaises(http.client.IncompleteRead):
        self.req(f"/local/file/internal/{DONGLE}/{LOG}/3/rlog.zst")
      self.assertIsNone(cache.lookup(cache.Key(DONGLE, LOG, 3, "rlog.zst")))
      self.assertFalse([p for p in os.listdir(CACHE) if p.startswith(cache.TMP) and "manifest" not in p])
    finally:
      Upstream.fail = False

  def test_file_rejects(self):
    for path in (f"/local/file/internal/{DONGLE}/{LOG}/0/..%2f..%2fx", f"/local/file/internal/{DONGLE}/{LOG}/0/passwd",
                 f"/local/file/nope/{DONGLE}/{LOG}/0/rlog.zst", f"/local/file/internal/zz/{LOG}/0/rlog.zst",
                 f"/local/file/local/{DONGLE}/..%2f..%2f/0/rlog.zst"):
      r, _ = self.req(path)
      self.assertIn(r.status, (403, 404), path)

  def test_local_source(self):
    d = Path(LOGS) / f"{LOG}--0"
    d.mkdir(exist_ok=True)
    (d / "qlog.zst").write_bytes(b"local qlog")
    res = self.json("/local/sources/resolve", {"route": ROUTE, "segs": [0], "kinds": ["qlog"], "sources": ["local"]})
    self.assertEqual(res["segments"]["0"]["qlog"]["source"], "local")
    r, data = self.req(f"/local/file/local/{DONGLE}/{LOG}/0/qlog.zst")
    self.assertEqual(data, b"local qlog")

  def test_prefetch_and_evict(self):
    for seg in (0, 1):
      cache.evict(cache.Key(DONGLE, LOG, seg, "qcamera.ts"))
    job = self.json("/local/cache/prefetch", {"route": ROUTE, "segs": [0, 1], "kinds": ["qcamera"], "sources": ONLY})
    j = jobs.get(job["id"])
    for _ in range(200):
      if j.state != "running":
        break
      time.sleep(0.02)
    self.assertEqual(j.state, "done", j.error)
    self.assertEqual(j.meta["segs"], {"0": "disk", "1": "disk"})
    stats = self.json(f"/local/cache?route={ROUTE}&segs=2")
    self.assertEqual(stats["route"]["segments"]["1"]["qcamera"]["size"], len(blob(1, "qcamera.ts")))
    out = self.json("/local/cache/evict", {"route": ROUTE, "segs": [0, 1], "kinds": ["qcamera"]})
    self.assertGreaterEqual(out["freed"], len(blob(0, "qcamera.ts")) + len(blob(1, "qcamera.ts")))
    self.assertIsNone(cache.lookup(cache.Key(DONGLE, LOG, 0, "qcamera.ts")))

  def test_fetch_for_modules(self):
    p = cache.fetch(DONGLE, LOG, 0, "qcamera.ts", sources=ONLY)
    self.assertEqual(p.read_bytes(), blob(0, "qcamera.ts"))

  def test_usage_running_total(self):
    cache._usage["at"] = -cache.RESCAN
    cache.evict(cache.Key(DONGLE, LOG, 2, "qcamera.ts"))
    cache.evict(cache.Key(DONGLE, LOG, 2, "qlog.zst"))
    walk = lambda: (sum(s for _, s, _ in cache.scan()), len(cache.scan()))  # noqa: E731
    self.assertEqual(cache.totals(), walk())
    cache.fetch(DONGLE, LOG, 2, "qcamera.ts", sources=ONLY)
    cache.fetch(DONGLE, LOG, 2, "qlog.zst", sources=ONLY)  # a log: also mirrored as url_file chunks
    self.assertEqual(cache.totals(), walk())
    cache.evict(cache.Key(DONGLE, LOG, 2, "qlog.zst"))
    self.assertEqual(cache.totals(), walk())
    t = time.monotonic()
    for _ in range(1000):
      cache.usage()
    self.assertLess((time.monotonic() - t) / 1000, 1e-4)  # polling never walks the directory

  def test_prune_lru(self):
    d = Path(tempfile.mkdtemp())
    old = config.cache_root
    config.cache_root = str(d) + "/"
    try:
      now = time.time()  # noqa: TID251  compared with file mtimes
      for i in range(5):
        (d / f"f{i}").write_bytes(b"x" * 1000)
        os.utime(d / f"f{i}", (now - 1000 + i, now - 1000 + i))
      (d / "recent").write_bytes(b"x" * 1000)
      out = cache.prune(3000)
      self.assertEqual(sorted(p.name for p in d.iterdir()), ["f3", "f4", "recent"])
      self.assertEqual(out["bytes"], 3000)
    finally:
      config.cache_root = old
      cache._usage["at"] = -cache.RESCAN

  def test_proxy(self):
    r, _ = self.req("/local/proxy?url=https://evil.example/")
    self.assertEqual(r.status, 403)
    r, _ = self.req("/local/proxy?url=http://169.254.169.254/latest/meta-data")
    self.assertEqual(r.status, 403)
    r, _ = self.req("/local/proxy?url=https://commadataci.blob.core.windows.net@evil.example/")
    self.assertEqual(r.status, 403)
    base = os.environ["DATA_ENDPOINT"]
    r, data = self.req(f"/local/proxy?url={base}{DONGLE}/{LOG}/0/qcamera.ts")
    self.assertEqual((r.status, data, r.getheader("Content-Type")), (200, blob(0, "qcamera.ts"), "application/octet-stream"))
    r, _ = self.req(f"/local/proxy?url={base}redirect")
    self.assertEqual(r.status, 403)  # the redirect target isn't allowed


if __name__ == "__main__":
  unittest.main()
