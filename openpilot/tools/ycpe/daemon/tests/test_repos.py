#!/usr/bin/env python3
import http.client
import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from openpilot.tools.ycpe.daemon import core
from openpilot.tools.ycpe.daemon.core import config
from openpilot.tools.ycpe.daemon.tests.test_security import DaemonCase

REPO = Path(tempfile.mkdtemp())
(REPO / "openpilot" / "cereal" / "include").mkdir(parents=True)
(REPO / "openpilot" / "cereal" / "log.capnp").write_text("@0xf3b1f17e25a4285b;\nstruct Event {}\n")
(REPO / "openpilot" / "cereal" / "include" / "c++.capnp").write_text("@0xbdf87d7bb8304e81;\n")
DBC = REPO / "opendbc_repo" / "opendbc" / "dbc"
DBC.mkdir(parents=True)
(DBC / "toyota_test.dbc").write_text('VERSION ""\n')
(REPO / "secret.dbc").write_text("outside")
subprocess.run(["git", "init", "-q", str(REPO)], check=True)
subprocess.run(["git", "-C", str(REPO), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "init"], check=True)


class TestRepos(DaemonCase):
  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    config.repo_roots = [str(REPO)]

  def json(self, path, **kw):
    r, data = self.req(path, **kw)
    self.assertEqual(r.status, 200, data)
    return json.loads(data)

  def test_list(self):
    [c] = self.json("/local/repos")
    self.assertEqual(c["path"], str(REPO.resolve()))
    self.assertEqual(c["git"]["subject"], "init")
    self.assertEqual(c["opendbc"]["dbc"], str(DBC.resolve()))

  def test_capnp(self):
    files = self.json("/local/repos/0/capnp")["files"]
    self.assertEqual(sorted(files), ["include/c++.capnp", "log.capnp"])

  def test_dbc_read_write(self):
    self.assertEqual([d["name"] for d in self.json("/local/repos/0/dbc")], ["toyota_test.dbc"])
    got = self.json("/local/repos/0/dbc/toyota_test.dbc")
    self.assertEqual(got["text"], 'VERSION ""\n')
    out = self.json(f"/local/repos/0/dbc/toyota_test.dbc?mtime={got['mtime']}", method="PUT", body=b'VERSION "2"\n')
    self.assertEqual((DBC / "toyota_test.dbc").read_text(), 'VERSION "2"\n')
    r, _ = self.req(f"/local/repos/0/dbc/toyota_test.dbc?mtime={got['mtime'] - 100}", method="PUT", body=b"x")
    self.assertEqual(r.status, 409)  # changed on disk since it was read
    self.json("/local/repos/0/dbc/new_car.dbc", method="PUT", body=b'VERSION ""\n')
    self.assertTrue((DBC / "new_car.dbc").is_file())
    self.assertFalse(out["generated"])

  def test_dbc_write_rejects(self):
    for name in ("..%2fsecret.dbc", "..%2f..%2fsecret.dbc", "x.txt", "%2etmp", "a%2fb.dbc", "..", "x.dbc%00"):
      r, _ = self.req(f"/local/repos/0/dbc/{name}", method="PUT", body=b"pwned")
      self.assertIn(r.status, (403, 404), name)
    self.assertEqual((REPO / "secret.dbc").read_text(), "outside")
    r, _ = self.req("/local/repos/0/dbc/toyota_test.dbc", method="PUT", body=b"x", token=False)
    self.assertEqual(r.status, 403)
    r, _ = self.req("/local/repos/0/dbc/toyota_test.dbc", method="PUT", body=b"x", headers={"Origin": "http://evil.example"})
    self.assertEqual(r.status, 403)
    r, _ = self.req("/local/repos/9/dbc/toyota_test.dbc", method="PUT", body=b"x")
    self.assertEqual(r.status, 404)

  def test_watch(self):
    c = http.client.HTTPConnection("127.0.0.1", config.port, timeout=10)
    c.request("GET", "/local/repos/watch", headers={"Host": f"localhost:{config.port}", core.TOKEN_HEADER: config.token})
    r = c.getresponse()
    time.sleep(1.2)  # the watcher's first snapshot
    p = DBC / "watched.dbc"
    p.write_text("a")
    os.utime(p, (time.time() + 5, time.time() + 5))  # noqa: TID251  a distinct mtime
    seen = b""
    while b"watched.dbc" not in seen:
      seen += r.read1(4096)
    c.close()
    self.assertIn(b'"kind":"dbc"', seen)


if __name__ == "__main__":
  unittest.main()
