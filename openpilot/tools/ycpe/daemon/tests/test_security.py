#!/usr/bin/env python3
import json
import os
import re
import socket
import tempfile
import threading
import unittest
from unittest import mock

from openpilot.tools.ycpe.daemon import core, jobs
from openpilot.tools.ycpe.daemon.core import File, config, confine, route
from openpilot.tools.ycpe.daemon.tests.common import DaemonCase


@route("GET", "/local/test/file/{path:path}")
def _test_file(req):
  return File(str(confine(os.path.join(_root, req.params["path"]), [_root])))


_root = tempfile.mkdtemp()
with open(os.path.join(_root, "data.bin"), "wb") as f:
  f.write(bytes(range(256)) * 100)
with open(os.path.join(os.path.dirname(_root), "secret.txt"), "w") as f:
  f.write("secret")


class TestSecurity(DaemonCase):
  def test_binds_loopback(self):
    self.assertEqual(self.srv.server_address[0], "127.0.0.1")

  def test_caps_public(self):
    r, data = self.req("/local/caps", token=False)
    self.assertEqual(r.status, 200)
    caps = json.loads(data)
    self.assertIn("auth", caps["modules"])
    self.assertIn("jobs", caps["modules"])
    self.assertRegex(caps["commit"]["sha"], r"^[0-9a-f]{40}$")

  def test_manifest_type(self):
    r, data = self.req("/manifest.webmanifest", token=False)
    self.assertEqual(r.getheader("Content-Type"), "application/manifest+json")
    self.assertEqual(json.loads(data)["display"], "standalone")

  def test_wrong_host(self):
    for path in ("/local/caps", "/local/auth", "/", "/index.html", "/api.js"):
      for host in (f"evil.example:{config.port}", "localhost:1", "localhost", f"127.0.0.1.nip.io:{config.port}"):
        r, _ = self.req(path, host=host)
        self.assertEqual(r.status, 403, (path, host))

  @unittest.skipUnless(hasattr(socket, "AF_NETLINK"), "linux only")
  def test_other_user(self):
    r, _ = self.req("/")
    self.assertEqual(r.status, 200)
    with mock.patch("os.getuid", return_value=os.getuid() + 1):
      for path in ("/", "/index.html", "/local/caps", "/local/auth"):
        r, data = self.req(path)
        self.assertEqual(r.status, 403, path)
        self.assertNotIn(config.token.encode(), data)

  def test_127_host_ok(self):
    r, _ = self.req("/local/caps", host=f"127.0.0.1:{config.port}")
    self.assertEqual(r.status, 200)

  def test_every_route(self):
    # every route refuses before it runs: no or a wrong token, another page, a cross-site fetch, a rebound host
    bad = [({"token": False}, "missing or wrong token"),
           ({"token": False, "headers": {core.TOKEN_HEADER: "x" * len(config.token)}}, "missing or wrong token"),
           ({"headers": {"Origin": "http://evil.example"}}, "bad origin"),
           ({"headers": {"Sec-Fetch-Site": "cross-site"}}, "cross-site"),
           ({"headers": {"Sec-Fetch-Site": "same-site"}}, "cross-site"),
           ({"host": f"evil.example:{config.port}"}, "bad host")]
    self.assertGreater(len(core._routes), 15)
    for method, rx, _, public, _ in core._routes:
      path = re.sub(r"\(\?P<\w+>[^)]*\)", "x", rx.pattern)[1:-2]
      for kw, error in bad:
        if public and error.endswith("token"):
          continue
        r, data = self.req(path, method=method, body=None if method == "GET" else b"{}", **kw)
        self.assertEqual((r.status, json.loads(data)["error"]), (403, error), (method, path, kw))

  def test_origins(self):
    for origin in ("http://evil.example", f"http://localhost:{config.port + 1}", "null", f"https://localhost:{config.port}"):
      r, _ = self.req("/local/repos", headers={"Origin": origin})
      self.assertEqual(r.status, 403, origin)
    r, _ = self.req("/local/repos", headers={"Origin": f"http://localhost:{config.port}"})
    self.assertEqual(r.status, 200)

  def test_token_param(self):
    r, _ = self.req(f"/local/repos?{core.TOKEN_PARAM}={config.token}", token=False)
    self.assertEqual(r.status, 200)

  def test_auth_no_leak(self):
    # the response carries the token (or null); it must not be printed here
    r, data = self.req("/local/auth")
    self.assertEqual(r.status, 200)
    self.assertIn("token", json.loads(data))

  def test_token_in_index(self):
    r, data = self.req("/", token=False)
    self.assertEqual(r.status, 200)
    self.assertTrue(f'<meta name="ycpe-daemon" content="{config.token}">'.encode() in data)  # assertTrue: no token in failure output

  def test_local_page_routes(self):
    # the app's own /local/<name>/<view> pages: a navigation gets index.html, an API call a JSON 404
    for headers in ({"Sec-Fetch-Mode": "navigate"}, {"Accept": "text/html,application/xhtml+xml"}):
      r, data = self.req("/local/demo-rlog/plot?url=x", headers=headers, token=False)
      self.assertEqual((r.status, r.getheader("Content-Type")), (200, "text/html; charset=utf-8"))
      self.assertTrue(b'name="ycpe-daemon"' in data)
    r, data = self.req("/local/demo-rlog/plot", headers={"Accept": "*/*", "Sec-Fetch-Mode": "cors"})
    self.assertEqual((r.status, json.loads(data)), (404, {"error": "Not Found"}))
    r, _ = self.req("/local/demo-rlog/plot", headers={"Sec-Fetch-Mode": "navigate"}, host="evil.example")
    self.assertEqual(r.status, 403)  # the host check still comes first
    r, _ = self.req("/local/repos/nope/dbc", headers={"Sec-Fetch-Mode": "navigate"})
    self.assertEqual(r.status, 404)  # a route's own 404 stays an API answer

  def test_traversal(self):
    for path in ("../secret.txt", "..%2fsecret.txt", "%2e%2e/secret.txt", "a/../../secret.txt", "/etc/passwd", "%2fetc%2fpasswd"):
      r, _ = self.req(f"/local/test/file/{path}")
      self.assertEqual(r.status, 403, path)
    link = os.path.join(_root, "link")
    if not os.path.exists(link):
      os.symlink(os.path.dirname(_root), link)
    r, _ = self.req("/local/test/file/link/secret.txt")
    self.assertEqual(r.status, 403)

  def test_no_framing(self):
    for path in ("/", "/local/caps"):
      r, _ = self.req(path, token=False)
      self.assertEqual(r.getheader("X-Frame-Options"), "DENY", path)
      self.assertEqual(r.getheader("Content-Security-Policy"), "frame-ancestors 'none'", path)

  def test_confine(self):
    self.assertEqual(str(confine(os.path.join(_root, "data.bin"), [_root])), os.path.realpath(os.path.join(_root, "data.bin")))
    with self.assertRaises(core.HTTPError):
      confine(_root + "-sibling/x", [_root])


class TestStreaming(DaemonCase):
  def test_range(self):
    full = bytes(range(256)) * 100
    r, data = self.req("/local/test/file/data.bin")
    self.assertEqual((r.status, data), (200, full))
    r, data = self.req("/local/test/file/data.bin", headers={"Range": "bytes=10-19"})
    self.assertEqual((r.status, data, r.getheader("Content-Range")), (206, full[10:20], f"bytes 10-19/{len(full)}"))
    r, data = self.req("/local/test/file/data.bin", headers={"Range": "bytes=-5"})
    self.assertEqual(data, full[-5:])
    r, data = self.req("/local/test/file/data.bin", headers={"Range": "bytes=25000-"})
    self.assertEqual(data, full[25000:])
    r, _ = self.req("/local/test/file/data.bin", headers={"Range": "bytes=999999-"})
    self.assertEqual(r.status, 416)

  def test_growing(self):
    path = os.path.join(_root, "growing.bin")
    open(path, "wb").close()
    done = threading.Event()

    @route("GET", "/local/test/growing")
    def _growing(req):
      return File(path, growing=lambda: not done.is_set())

    def write():
      with open(path, "ab") as f:
        for i in range(5):
          f.write(bytes([i]) * 1000)
          f.flush()
          threading.Event().wait(0.05)
      done.set()
    threading.Thread(target=write).start()
    r, data = self.req("/local/test/growing")
    self.assertEqual(r.status, 200)
    self.assertEqual(data, b"".join(bytes([i]) * 1000 for i in range(5)))

  def test_jobs(self):
    gate = threading.Event()

    def work(job):
      for i in range(100):
        job.progress(i, 100)
        if gate.wait(0.01):
          job.check()
      return {"ok": True}
    j = jobs.start("test", work)
    self.assertEqual(j.state, "running")
    r, data = self.req(f"/local/jobs/{j.id}/cancel", method="POST")
    gate.set()
    self.assertTrue(j.finished.wait(1))
    self.assertEqual(j.state, "cancelled")
    j2 = jobs.start("test", lambda job: {"n": 1})
    self.assertTrue(j2.finished.wait(1))
    self.assertEqual(j2.result, {"n": 1})

  def test_job_events(self):
    r = self.connect("/local/jobs/events")
    self.assertEqual(r.getheader("Content-Type"), "text/event-stream")
    jobs.start("evt", lambda job: 42)
    self.read_until(r, b'"result":42')


if __name__ == "__main__":
  unittest.main()
