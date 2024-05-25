import os
import pathlib
import shutil
import signal
import stat
import subprocess
import tempfile
import time
import pytest

from openpilot.common.params import Params
from openpilot.system.manager.process import ManagerProcess
from openpilot.selfdrive.test.helpers import processes_context


def get_consistent_flag(path: str) -> bool:
  consistent_file = pathlib.Path(os.path.join(path, ".overlay_consistent"))
  return consistent_file.is_file()


def run(args, **kwargs):
  return subprocess.check_output(args, **kwargs)


def update_release(directory, name, version, agnos_version, release_notes):
  with open(directory / "RELEASES.md", "w") as f:
    f.write(release_notes)

  (directory / "common").mkdir(exist_ok=True)

  with open(directory / "common" / "version.h", "w") as f:
    f.write(f'#define COMMA_VERSION "{version}"')

  launch_env = directory / "launch_env.sh"
  with open(launch_env, "w") as f:
    f.write(f'export AGNOS_VERSION="{agnos_version}"')

  st = os.stat(launch_env)
  os.chmod(launch_env, st.st_mode | stat.S_IEXEC)

  test_symlink = directory / "test_symlink"
  if not os.path.exists(str(test_symlink)):
    os.symlink("common/version.h", test_symlink)


def get_version(path: str) -> str:
  with open(os.path.join(path, "common", "version.h")) as f:
    return f.read().split('"')[1]


@pytest.mark.slow # TODO: can we test overlayfs in GHA?
class TestBaseUpdate:
  @classmethod
  def setup_class(cls):
    if "Base" in cls.__name__:
      pytest.skip()

  def setup_method(self):
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

    os.environ["UPDATER_STAGING_ROOT"] = str(self.staging_root)
    os.environ["UPDATER_LOCK_FILE"] = str(self.mock_update_path / "safe_staging_overlay.lock")

    self.MOCK_RELEASES = {
      "release3": ("0.1.2", "1.2", "0.1.2 release notes"),
      "master": ("0.1.3", "1.2", "0.1.3 release notes"),
    }

  @pytest.fixture(autouse=True)
  def mock_basedir(self, mocker):
    mocker.patch("openpilot.common.basedir.BASEDIR", self.basedir)

  def set_target_branch(self, branch):
    self.params.put("UpdaterTargetBranch", branch)

  def setup_basedir_release(self, release):
    self.params = Params()
    self.set_target_branch(release)

  def update_remote_release(self, release):
    raise NotImplementedError("")

  def setup_remote_release(self, release):
    raise NotImplementedError("")

  def additional_context(self):
    raise NotImplementedError("")

  def teardown_method(self):
    try:
      run(["sudo", "umount", "-l", str(self.staging_root / "merged")])
      run(["sudo", "umount", "-l", self.tmpdir])
      shutil.rmtree(self.tmpdir)
    except Exception:
      print("cleanup failed...")

  def wait_for_condition(self, condition, timeout=12):
    start = time.monotonic()
    while True:
      waited = time.monotonic() - start
      if condition():
        print(f"waited {waited}s for condition ")
        return waited

      if waited > timeout:
        raise TimeoutError("timed out waiting for condition")

      time.sleep(1)

  def _test_finalized_update(self, branch, version, agnos_version, release_notes):
    assert get_version(str(self.staging_root / "finalized")) == version
    assert get_consistent_flag(str(self.staging_root / "finalized"))
    assert os.access(str(self.staging_root / "finalized" / "launch_env.sh"), os.X_OK)

    with open(self.staging_root / "finalized" / "test_symlink") as f:
      assert version in f.read()

class ParamsBaseUpdateTest(TestBaseUpdate):
  def _test_finalized_update(self, branch, version, agnos_version, release_notes):
    assert self.params.get("UpdaterNewDescription", encoding="utf-8").startswith(f"{version} / {branch}")
    assert self.params.get("UpdaterNewReleaseNotes", encoding="utf-8") == f"<p>{release_notes}</p>\n"
    super()._test_finalized_update(branch, version, agnos_version, release_notes)

  def send_check_for_updates_signal(self, updated: ManagerProcess):
    updated.signal(signal.SIGUSR1.value)

  def send_download_signal(self, updated: ManagerProcess):
    updated.signal(signal.SIGHUP.value)

  def _test_params(self, branch, fetch_available, update_available):
    assert self.params.get("UpdaterTargetBranch", encoding="utf-8") == branch
    assert self.params.get_bool("UpdaterFetchAvailable") == fetch_available
    assert self.params.get_bool("UpdateAvailable") == update_available

  def wait_for_idle(self):
    self.wait_for_condition(lambda: self.params.get("UpdaterState", encoding="utf-8") == "idle")

  def wait_for_failed(self):
    self.wait_for_condition(lambda: self.params.get("UpdateFailedCount", encoding="utf-8") is not None and \
                                              int(self.params.get("UpdateFailedCount", encoding="utf-8")) > 0)

  def wait_for_fetch_available(self):
    self.wait_for_condition(lambda: self.params.get_bool("UpdaterFetchAvailable"))

  def wait_for_update_available(self):
    self.wait_for_condition(lambda: self.params.get_bool("UpdateAvailable"))

  def test_no_update(self):
    # Start on release3, ensure we don't fetch any updates
    self.setup_remote_release("release3")
    self.setup_basedir_release("release3")

    with self.additional_context(), processes_context(["updated"]) as [updated]:
      self._test_params("release3", False, False)
      self.wait_for_idle()
      self._test_params("release3", False, False)

      self.send_check_for_updates_signal(updated)

      self.wait_for_idle()

      self._test_params("release3", False, False)

  def test_new_release(self):
    # Start on release3, simulate a release3 commit, ensure we fetch that update properly
    self.setup_remote_release("release3")
    self.setup_basedir_release("release3")

    with self.additional_context(), processes_context(["updated"]) as [updated]:
      self._test_params("release3", False, False)
      self.wait_for_idle()
      self._test_params("release3", False, False)

      self.MOCK_RELEASES["release3"] = ("0.1.3", "1.2", "0.1.3 release notes")
      self.update_remote_release("release3")

      self.send_check_for_updates_signal(updated)

      self.wait_for_fetch_available()

      self._test_params("release3", True, False)

      self.send_download_signal(updated)

      self.wait_for_update_available()

      self._test_params("release3", False, True)
      self._test_finalized_update("release3", *self.MOCK_RELEASES["release3"])

  def test_switch_branches(self):
    # Start on release3, request to switch to master manually, ensure we switched
    self.setup_remote_release("release3")
    self.setup_remote_release("master")
    self.setup_basedir_release("release3")

    with self.additional_context(), processes_context(["updated"]) as [updated]:
      self._test_params("release3", False, False)
      self.wait_for_idle()
      self._test_params("release3", False, False)

      self.set_target_branch("master")
      self.send_check_for_updates_signal(updated)

      self.wait_for_fetch_available()

      self._test_params("master", True, False)

      self.send_download_signal(updated)

      self.wait_for_update_available()

      self._test_params("master", False, True)
      self._test_finalized_update("master", *self.MOCK_RELEASES["master"])

  def test_agnos_update(self, mocker):
    # Start on release3, push an update with an agnos change
    self.setup_remote_release("release3")
    self.setup_basedir_release("release3")

    with self.additional_context(), processes_context(["updated"]) as [updated]:
      mocker.patch("openpilot.system.hardware.AGNOS", "True")
      mocker.patch("openpilot.system.hardware.tici.hardware.Tici.get_os_version", "1.2")
      mocker.patch("openpilot.system.hardware.tici.agnos.get_target_slot_number")
      mocker.patch("openpilot.system.hardware.tici.agnos.flash_agnos_update")

      self._test_params("release3", False, False)
      self.wait_for_idle()
      self._test_params("release3", False, False)

      self.MOCK_RELEASES["release3"] = ("0.1.3", "1.3", "0.1.3 release notes")
      self.update_remote_release("release3")

      self.send_check_for_updates_signal(updated)

      self.wait_for_fetch_available()

      self._test_params("release3", True, False)

      self.send_download_signal(updated)

      self.wait_for_update_available()

      self._test_params("release3", False, True)
      self._test_finalized_update("release3", *self.MOCK_RELEASES["release3"])
