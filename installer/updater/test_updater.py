#!/usr/bin/env python3
import os
import subprocess
import tempfile
import time
import unittest

from common.basedir import BASEDIR

UPDATER_PATH = os.path.join(BASEDIR, "installer/updater")
UPDATER = os.path.join(UPDATER_PATH, "updater")
UPDATE_MANIFEST = os.path.join(UPDATER_PATH, "update.json")


class TestUpdater(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    # test that the updater builds
    cls.assertTrue(f"cd {UPDATER_PATH} && make clean && make", "updater failed to build")

    # restore the checked-in version, since that's what actually runs on devices
    os.system(f"git reset --hard {UPDATER_PATH}")

  def _assert_ok(self, cmd, msg=None):
    self.assertTrue(os.system(cmd) == 0, msg)

  def _assert_fails(self, cmd):
    self.assertFalse(os.system(cmd) == 0)

  def test_background_download(self):
    print(f"{UPDATER} bgcache 'file://{UPDATE_MANIFEST}'")
    self._assert_ok(f"{UPDATER} bgcache 'file://{UPDATE_MANIFEST}'")

  def test_background_download_bad_manifest(self):
    # update with bad manifest should fail
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
      f.write("{}")
      self._assert_fails(f"{UPDATER} bgcache 'file://{f.name}'")

  # simple test that the updater doesn't crash in UI mode
  def test_ui_init(self):
    with subprocess.Popen(UPDATER) as proc:
      time.sleep(5)
      self.assertTrue(proc.poll() is None)
      proc.terminate()

if __name__ == "__main__":
  unittest.main()
