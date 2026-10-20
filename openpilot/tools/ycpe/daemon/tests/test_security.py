#!/usr/bin/env python3
import http.client
import json
import os
import socket
import tempfile
import threading
import unittest
from unittest import mock

from openpilot.tools.ycpe.daemon import core, jobs, server
from openpilot.tools.ycpe.daemon.core import File, config, confine, route


@route("GET", "/local/test/file/{path:path}")
def _test_file(req):
  return File(str(confine(os.path.join(_root, req.params["path"]), [_root])))


_root = tempfile.mkdtemp()


class DaemonCase(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    config.log_roots = [_root]
    server.Handler.log_message = lambda *a: None
    cls.srv = server.make_server(0)
    config.port = cls.srv.server_address[1]
    threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
    with open(os.path.join(_root, "data.bin"), "wb") as f:
      f.write(bytes(range(256)) * 100)
    with open(os.path.join(os.path.dirname(_root), "secret.txt"), "w") as f:
      f.write("secret")

  @classmethod
  def tearDownClass(cls):
    cls.srv.shutdown()
    cls.srv.server_close()

  def req(self, path, method="GET", host=None, headers=None, token=True, body=None):
    c = http.client.HTTPConnection("127.0.0.1", config.port, timeout=10)
    h = {"Host": host or f"localhost:{config.port}", **(headers or {})}
    if token:
      h[core.TOKEN_HEADER] = config.token
    c.request(method, path, body=body, headers=h)
    r = c.getresponse()
    data = r.read()
    c.close()
    return r, data


class TestSecurity(DaemonCase):
  def test_binds_loopback(self):
    self.assertEqual(self.srv.server_address[0], "127.0.0.1")

  def test_caps_public(self):
    r, data = self.req("/local/caps", token=False)
    self.assertEqual(r.status, 200)
    caps = json.loads(data)
    self.assertIn("auth", caps["modules"])
    self.assertIn("jobs", caps["modules"])

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

  def test_wrong_origin(self):
    for origin in ("http://evil.example", f"http://localhost:{config.port + 1}", "null", f"https://localhost:{config.port}"):
      for path in ("/local/caps", "/local/auth", "/local/jobs"):
        r, _ = self.req(path, headers={"Origin": origin})
        self.assertEqual(r.status, 403, (path, origin))
    r, _ = self.req("/local/jobs", headers={"Origin": f"http://localhost:{config.port}"})
    self.assertEqual(r.status, 200)

  def test_cross_site_fetch(self):
    r, _ = self.req("/local/jobs", headers={"Sec-Fetch-Site": "cross-site"})
    self.assertEqual(r.status, 403)
    r, _ = self.req("/local/jobs", headers={"Sec-Fetch-Site": "same-site"})
    self.assertEqual(r.status, 403)

  def test_missing_token(self):
    for path in ("/local/auth", "/local/jobs", "/local/test/file/data.bin"):
      r, _ = self.req(path, token=False)
      self.assertEqual(r.status, 403, path)
      r, _ = self.req(path, token=False, headers={core.TOKEN_HEADER: "x" * len(config.token)})
      self.assertEqual(r.status, 403, path)

  def test_token_param(self):
    r, _ = self.req(f"/local/jobs?{core.TOKEN_PARAM}={config.token}", token=False)
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
    r, _ = self.req("/local/jobs/nope", headers={"Sec-Fetch-Mode": "navigate"})
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
    r, data = self.req(f"/local/jobs/{j.id}")
    self.assertEqual(json.loads(data)["state"], "running")
    r, data = self.req(f"/local/jobs/{j.id}/cancel", method="POST")
    gate.set()
    for _ in range(100):
      if j.state != "running":
        break
      threading.Event().wait(0.01)
    self.assertEqual(j.state, "cancelled")
    j2 = jobs.start("test", lambda job: {"n": 1})
    for _ in range(100):
      if j2.state != "running":
        break
      threading.Event().wait(0.01)
    self.assertEqual(j2.result, {"n": 1})

  def test_job_events(self):
    c = http.client.HTTPConnection("127.0.0.1", config.port, timeout=10)
    c.request("GET", "/local/jobs/events", headers={"Host": f"localhost:{config.port}", core.TOKEN_HEADER: config.token})
    r = c.getresponse()
    self.assertEqual(r.getheader("Content-Type"), "text/event-stream")
    jobs.start("evt", lambda job: 42)
    seen = b""
    while b'"result":42' not in seen:
      seen += r.read1(4096)
    c.close()


if __name__ == "__main__":
  unittest.main()
