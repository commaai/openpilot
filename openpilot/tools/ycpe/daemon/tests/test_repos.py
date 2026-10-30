#!/usr/bin/env python3
import tempfile
import unittest
from pathlib import Path

from openpilot.tools.ycpe.daemon.core import config
from openpilot.tools.ycpe.daemon.tests.common import DaemonCase

REPO = Path(tempfile.mkdtemp())
DBC = REPO / "opendbc_repo" / "opendbc" / "dbc"
DBC.mkdir(parents=True)
(DBC / "toyota_test.dbc").write_text('VERSION ""\n')
(REPO / "secret.dbc").write_text("outside")
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
    self.assertEqual(c["opendbc"]["dbc"], str(DBC.resolve()))

  def test_dbc_read_write(self):
    self.assertEqual([d["name"] for d in self.json("/local/repos/0/dbc")], ["toyota_test.dbc"])
    got = self.json("/local/repos/0/dbc/toyota_test.dbc")
    self.assertEqual(got["text"], 'VERSION ""\n')
    out = self.json(f"/local/repos/0/dbc/toyota_test.dbc?mtime={got['mtime']}", method="PUT", body=b'VERSION "2"\n')
    self.assertEqual((DBC / "toyota_test.dbc").read_text(), 'VERSION "2"\n')
    stale = self.status(f"/local/repos/0/dbc/toyota_test.dbc?mtime={got['mtime'] - 100}", method="PUT", body=b"x")
    self.assertEqual(stale, 409)  # changed on disk since it was read
    self.json("/local/repos/0/dbc/new_car.dbc", method="PUT", body=b'VERSION ""\n')
    self.assertTrue((DBC / "new_car.dbc").is_file())
    self.assertFalse(out["generated"])

  def test_dbc_write_rejects(self):
    for name in ("..%2fsecret.dbc", "..%2f..%2fsecret.dbc", "x.txt", "%2etmp", "a%2fb.dbc", "..", "x.dbc%00", "x.dbc%0A"):
      self.assertIn(self.status(f"/local/repos/0/dbc/{name}", method="PUT", body=b"pwned"), (403, 404), name)
    self.assertEqual((REPO / "secret.dbc").read_text(), "outside")
    self.assertEqual(self.status("/local/repos/9/dbc/toyota_test.dbc", method="PUT", body=b"x"), 404)
    # never the installed opendbc package, which may be another checkout's
    self.assertEqual(self.status("/local/repos/1/dbc/toyota_test.dbc", method="PUT", body=b"x"), 404)


if __name__ == "__main__":
  unittest.main()
