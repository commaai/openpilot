#!/usr/bin/env python3
import json
import os
import unittest
import subprocess
import tempfile

from common.basedir import BASEDIR

UPDATER_PATH = os.path.join(BASEDIR, "installer/updater")
UPDATER = os.path.join(UPDATER_PATH, "updater")
UPDATE_MANIFEST = os.path.join(UPDATER_PATH, "update.json")


class TestUpdater(unittest.TestCase):

  def _assert_ok(self, cmd):
    self.assertTrue(os.system(cmd) == 0)

  def _assert_fails(self, cmd):
    self.assertFalse(os.system(cmd) == 0)

  def test_build(self):
    self._assert_ok(f"cd {UPDATER_PATH} && make clean && make")

  # test background download
  def test_background_download(self):
    print("testing background download")

    # actual manifest
    print(f"{UPDATER} bgcache 'file://{UPDATE_MANIFEST}'")
    self._assert_ok(f"{UPDATER} bgcache 'file://{UPDATE_MANIFEST}'")

    # bad manifest
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
      f.write("{}")
      self._assert_fails(f"{UPDATER} bgcache 'file://{f.name}'")


if __name__ == "__main__":
  unittest.main()
