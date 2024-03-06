import os
import pathlib
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest import mock

import pytest


from openpilot.selfdrive.test.helpers import processes_context
from openpilot.common.params import Params


def run(args, **kwargs):
  return subprocess.run(args, **kwargs, check=True)


def update_release(directory, name, version, release_notes):
  with open(directory / "RELEASES.md", "w") as f:
    f.write(release_notes)

  (directory / "common").mkdir(exist_ok=True)

  with open(directory / "common" / "version.h", "w") as f:
    f.write(f'#define COMMA_VERSION "{version}"')

  run(["git", "add", "."], cwd=directory)
  run(["git", "commit", "-m", f"openpilot release {version}"], cwd=directory)


@pytest.mark.slow # TODO: can we test overlayfs in GHA?
class TestUpdateD(unittest.TestCase):
  def setUp(self):
    self.tmpdir = tempfile.mkdtemp()

    run(["sudo", "mount", "-t", "tmpfs", "tmpfs", self.tmpdir]) # overlayfs doesn't work inside of docker unless this is a tmpfs

    self.mock_update_path = pathlib.Path(self.tmpdir)

    self.params = Params()

    self.basedir = self.mock_update_path / "openpilot"
    self.basedir.mkdir()

    self.staging_root = self.mock_update_path / "safe_staging"
    self.staging_root.mkdir()

    self.remote_dir = self.mock_update_path / "remote"
    self.remote_dir.mkdir()

    mock.patch("openpilot.common.basedir.BASEDIR", self.basedir).start()

    os.environ["UPDATER_STAGING_ROOT"] = str(self.staging_root)
    os.environ["UPDATER_LOCK_FILE"] = str(self.mock_update_path / "safe_staging_overlay.lock")

    self.MOCK_RELEASES = {
      "release3": ("0.1.2", "0.1.2 release notes"),
      "master": ("0.1.3", "0.1.3 release notes"),
    }

  def set_target_branch(self, branch):
    self.params.put("UpdaterTargetBranch", branch)

  def setup_basedir_release(self, release):
    self.params = Params()
    self.set_target_branch(release)
    run(["git", "clone", "-b", release, self.remote_dir, self.basedir])

  def update_remote_release(self, release):
    update_release(self.remote_dir, release, *self.MOCK_RELEASES[release])

  def setup_remote_release(self, release):
    run(["git", "init"], cwd=self.remote_dir)
    run(["git", "checkout", "-b", release], cwd=self.remote_dir)
    self.update_remote_release(release)

  def tearDown(self):
    mock.patch.stopall()
    run(["sudo", "umount", "-l", str(self.staging_root / "merged")])
    run(["sudo", "umount", "-l", self.tmpdir])
    shutil.rmtree(self.tmpdir)

  def send_check_for_updates_signal(self):
    subprocess.run(["pkill", "-SIGUSR1", "-f", "selfdrive.updated.updated"], check=False)

  def send_download_signal(self):
    subprocess.run(["pkill", "-SIGHUP", "-f", "selfdrive.updated.updated"], check=False)

  def _test_params(self, branch, fetch_available, update_available):
    self.assertEqual(self.params.get("UpdaterTargetBranch", encoding="utf-8"), branch)
    self.assertEqual(self.params.get_bool("UpdaterFetchAvailable"), fetch_available)
    self.assertEqual(self.params.get_bool("UpdateAvailable"), update_available)

  def _test_update_params(self, branch, version, release_notes):
    self.assertTrue(self.params.get("UpdaterNewDescription", encoding="utf-8").startswith(f"{version} / {branch}"))
    self.assertEqual(self.params.get("UpdaterNewReleaseNotes", encoding="utf-8"), f"<p>{release_notes}</p>\n")

  def test_new_release(self):
    self.setup_remote_release("release3")
    self.setup_basedir_release("release3")

    with processes_context(["updated"]):
      self._test_params("release3", False, False)
      time.sleep(1)
      self._test_params("release3", False, False)

      self.MOCK_RELEASES["release3"] = ("0.1.3", "0.1.3 release notes")
      self.update_remote_release("release3")

      self.send_check_for_updates_signal()

      time.sleep(3)

      self._test_params("release3", True, False)

      self.send_download_signal()

      time.sleep(4)

      self._test_params("release3", False, True)
      self._test_update_params("release3", *self.MOCK_RELEASES["release3"])

