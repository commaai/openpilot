#!/usr/bin/env python3
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from openpilot.tools.ycpe.daemon.core import config
from openpilot.tools.ycpe.daemon.tests.common import DaemonCase

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
BARE = Path(tempfile.mkdtemp())  # a checkout without the opendbc submodule
(BARE / "opendbc_repo" / "opendbc" / "dbc").mkdir(parents=True)


class TestRepos(DaemonCase):
  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    config.repo_roots = [str(REPO), str(BARE)]

  def test_list(self):
    c, bare = self.json("/local/repos")
    self.assertIsNone(bare["opendbc"])
    self.assertEqual(c["name"], REPO.name)
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    self.assertEqual((c["git"]["commit"], c["git"]["dirty"]), (commit[:10], False))
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
    for name in ("..%2fsecret.dbc", "..%2f..%2fsecret.dbc", "x.txt", "%2etmp", "a%2fb.dbc", "..", "x.dbc%00", "x.dbc%0A"):
      r, _ = self.req(f"/local/repos/0/dbc/{name}", method="PUT", body=b"pwned")
      self.assertIn(r.status, (403, 404), name)
    self.assertEqual((REPO / "secret.dbc").read_text(), "outside")
    r, _ = self.req("/local/repos/9/dbc/toyota_test.dbc", method="PUT", body=b"x")
    self.assertEqual(r.status, 404)
    r, _ = self.req("/local/repos/1/dbc/toyota_test.dbc", method="PUT", body=b"x")
    self.assertEqual(r.status, 404)  # never the installed opendbc package, which may be another checkout's

  def test_watch(self):
    r = self.connect("/local/repos/watch")
    time.sleep(1.2)  # the watcher's first snapshot
    p = REPO / "openpilot" / "cereal" / "watched.capnp"
    p.write_text("a")
    os.utime(p, (time.time() + 5, time.time() + 5))  # noqa: TID251  a distinct mtime
    self.assertIn(b'"kind":"capnp"', self.read_until(r, b"watched.capnp"))


if __name__ == "__main__":
  unittest.main()
