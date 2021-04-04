#!/usr/bin/env python3
import os
import shutil
import subprocess
import tempfile
import time
import unittest

from common.basedir import BASEDIR

UPDATER_PATH = os.path.join(BASEDIR, "selfdrive/ui/qt/setup")
UPDATER = os.path.join(UPDATER_PATH, "updater")
UPDATE_MANIFEST = os.path.join(UPDATER_PATH, "update.json")
UPDATE_DIR = "/data/neoupdate"

class TestUpdater(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    pass

  def setUp(self):
    self._clear_dir()

  def tearDown(self):
    self._clear_dir()

  def _clear_dir(self):
    if os.path.isdir(UPDATE_DIR):
      shutil.rmtree(UPDATE_DIR)

  def _assert_ok(self, cmd, msg=None):
    self.assertTrue(os.system(cmd) == 0, msg)

  def _assert_fails(self, cmd):
    self.assertFalse(os.system(cmd) == 0)

  def test_background_download(self):
    self._assert_ok(f"{UPDATER} bgcache 'file://{UPDATE_MANIFEST}'")

  def test_background_download_bad_manifest(self):
    # update with bad manifest should fail
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
      f.write("{}")
      self._assert_fails(f"{UPDATER} bgcache 'file://{f.name}'")

  def test_cache_resume(self):
    self._assert_ok(f"{UPDATER} bgcache 'file://{UPDATE_MANIFEST}'")
    # a full download takes >1m, but resuming from fully cached should only be a few seconds
    start_time = time.monotonic()
    self._assert_ok(f"{UPDATER} bgcache 'file://{UPDATE_MANIFEST}'")
    self.assertLess(time.monotonic() - start_time, 10)

  # make sure we can recover from corrupt downloads
  def test_recover_from_corrupt(self):
    # download the whole update
    self._assert_ok(f"{UPDATER} bgcache 'file://{UPDATE_MANIFEST}'")

    # write some random bytes
    for f in os.listdir(UPDATE_DIR):
      with open(os.path.join(UPDATE_DIR, f), "ab") as f:
        f.write(b"\xab"*20)

    # this attempt should fail, then it unlinks
    self._assert_fails(f"{UPDATER} bgcache 'file://{UPDATE_MANIFEST}'")

    # now it should pass
    self._assert_ok(f"{UPDATER} bgcache 'file://{UPDATE_MANIFEST}'")

  # simple test that the updater doesn't crash in UI mode
  def test_ui_init(self):
    with subprocess.Popen(UPDATER) as proc:
      time.sleep(5)
      self.assertTrue(proc.poll() is None)
      proc.terminate()

if __name__ == "__main__":
  unittest.main()
