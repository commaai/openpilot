import contextlib
import dataclasses
import http
import json
import os
import pathlib
import stat
import tempfile
import time
from unittest import mock
import unittest

from openpilot.common.params import Params

from openpilot.common.run import run_cmd
from openpilot.selfdrive.manager.process import PythonProcess
from openpilot.selfdrive.manager.process_config import only_offroad
from openpilot.selfdrive.test.helpers import DirectoryHttpServer, http_server_context, processes_context
from openpilot.system.hardware import PC
from openpilot.system.hardware.tici.agnos import get_raw_hash
from openpilot.system.updated.casync.common import create_build_metadata_file, create_casync_from_file, create_casync_release
from openpilot.system.updated.common import get_valid_flag
from openpilot.system.version import BuildMetadata, OpenpilotMetadata, get_version


def create_remote_response(channel, build_metadata, entries: list[dict], casync_base: str):

  for entry in entries:
    entry["casync"]["caibx"] = os.path.join(casync_base, os.path.basename(entry["casync"]["caibx"]))

  return {
    "build_metadata": dataclasses.asdict(build_metadata),
    "manifest": entries
  }


@contextlib.contextmanager
def fake_ab(target):
  def get_current_ab_slot():
    return target

  def get_target_ab_slot():
    return "_b" if target == "_a" else "_a"

  def prepare_target_ab_slot():
    pass

  with mock.patch("openpilot.system.hardware.HARDWARE.get_current_ab_slot", get_current_ab_slot), \
      mock.patch("openpilot.system.hardware.HARDWARE.get_target_ab_slot", get_target_ab_slot), \
      mock.patch("openpilot.system.hardware.HARDWARE.prepare_target_ab_slot", prepare_target_ab_slot):
    yield


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


def create_mock_build_metadata(channel, version, agnos_version, release_notes):
  commit = str(hash((version, agnos_version, release_notes)))  # simulated commit
  return BuildMetadata(channel, OpenpilotMetadata(version, release_notes, commit, "https://github.com/commaai/openpilot.git", "2024", "release", False))


def create_casync_files(dirname, channel, version, agnos_version, release_notes):
  create_build_metadata_file(pathlib.Path(dirname), create_mock_build_metadata(channel, version, agnos_version, release_notes), channel)


def OpenpilotChannelMockAPI(release_manifests: dict[str, list[dict]], mock_releases, casync_base):
  class Handler(http.server.BaseHTTPRequestHandler):
    API_BAD_RESPONSE = False
    API_NO_RESPONSE = False

    def do_GET(self):
      if self.API_NO_RESPONSE:
        return

      if self.API_BAD_RESPONSE:
        self.send_response(500, "")
        return

      if self.path == "/v1/openpilot/channels":
        response = list(release_manifests.keys())
      else:
        channel = self.path.split("/")[-1]
        build_metadata = create_mock_build_metadata(channel, *mock_releases[channel])

        response = create_remote_response(channel, build_metadata, release_manifests[channel], casync_base)

      response = json.dumps(response)

      self.send_response(200)
      self.send_header('Content-Type', 'application/json')
      self.end_headers()
      self.wfile.write(response.encode(encoding='utf_8'))

  return Handler


def create_virtual_agnos_manifest(mock_update_path: pathlib.Path, agnos_version: str) -> list[dict]:
  agnos_bin_file = mock_update_path / "agnos.bin"

  data = agnos_version.encode("utf-8")

  with open(agnos_bin_file, "wb") as f:
    f.write(data)

  caibx_file = create_casync_from_file(agnos_bin_file, mock_update_path / "casync", "agnos.caibx")

  size = len(data)
  raw_hash = get_raw_hash(str(agnos_bin_file), size)

  return [
    {
      "name": "system",
      "type": "partition",
      "path": f"{mock_update_path}/system_a",
      "ab": True,
      "casync": {
        "caibx": caibx_file.name
      },
      "size": size,
      "hash_raw": raw_hash,
      "full_check": True
    },
    {
      "name": "boot",
      "type": "partition",
      "path": f"{mock_update_path}/boot_a",
      "ab": True,
      "casync": {
        "caibx": caibx_file.name
      },
      "size": size,
      "hash_raw": raw_hash,
      "full_check": False
    },
  ]


class TestUpdated(unittest.TestCase):
  def setUp(self):
    self.tmpdir = tempfile.mkdtemp()

    self.mock_update_path = pathlib.Path(self.tmpdir)

    self.params = Params()

    self.userdata = self.mock_update_path / "userdata"
    self.userdata.mkdir()

    self.basedir = self.userdata / "openpilot"
    self.basedir.mkdir()

    self.finalized = self.userdata / "finalized"
    self.finalized.mkdir()

    self.remote_dir = self.mock_update_path / "remote"
    self.remote_dir.mkdir()

    os.environ["USERDATA_DIR"] = str(self.userdata)

    mock.patch("openpilot.common.basedir.BASEDIR", self.basedir).start()

    self.MOCK_RELEASES = {
      "release3": ("0.1.2", "1.2", "0.1.2 release notes"),
      "master": ("0.1.3", "1.2", "0.1.3 release notes"),
    }

    self.casync_dir = self.mock_update_path / "casync"
    self.casync_dir.mkdir()
    self.release_manifests = {}
    os.environ["UPDATE_DELAY"] = "1"

    self.system_a = self.mock_update_path / "system_a"
    self.system_b = self.mock_update_path / "system_b"

    self.system_a.write_bytes(b"1.2")
    self.system_b.write_bytes(b"1.2")

    self.boot_a = self.mock_update_path / "boot_a"
    self.boot_b = self.mock_update_path / "boot_b"

    self.boot_a.write_bytes(b"1.2")
    self.boot_b.write_bytes(b"1.2")

  def set_target_branch(self, branch):
    self.params.put("UpdaterTargetBranch", branch)

  def setup_basedir_release(self, release):
    self.params = Params()
    self.set_target_branch(release)
    update_release(self.basedir, release, *self.MOCK_RELEASES[release])
    create_casync_files(self.basedir, release, *self.MOCK_RELEASES[release])

  def wait_for_condition(self, condition, timeout=4):
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
    finalized_dir = self.userdata / "finalized"
    self.assertTrue(os.path.exists(finalized_dir / "openpilot" / "build.json"))
    self.assertEqual(get_version(str(finalized_dir / "openpilot")), version)
    self.assertEqual(get_valid_flag(str(finalized_dir)), True)
    self.assertTrue(os.access(str(finalized_dir / "openpilot" / "launch_env.sh"), os.X_OK))

    with open(finalized_dir / "openpilot" / "test_symlink") as f:
      self.assertIn(version, f.read())

  def update_remote_release(self, release):
    update_release(self.remote_dir, release, *self.MOCK_RELEASES[release])
    create_casync_files(self.remote_dir, release, *self.MOCK_RELEASES[release])

    digest, caibx_file = create_casync_release(self.remote_dir, self.casync_dir, release)
    self.release_manifests[release] = [
      {
        "type": "path_tarred",
        "path": f"{self.mock_update_path}/userdata/openpilot",
        "casync": {
          "caibx": caibx_file.name,
        }
    }]
    self.release_manifests[release] += create_virtual_agnos_manifest(self.mock_update_path, self.MOCK_RELEASES[release][1])

  def setup_remote_release(self, release):
    self.update_remote_release(release)

  @contextlib.contextmanager
  def additional_context(self):
    self.directory_handler = DirectoryHttpServer(self.casync_dir)
    with http_server_context(self.directory_handler) as (casync_host, casync_port):
      casync_base = f"http://{casync_host}:{casync_port}"

      self.api_handler = OpenpilotChannelMockAPI(self.release_manifests, self.MOCK_RELEASES, casync_base)
      with http_server_context(self.api_handler) as (api_host, api_port):
        api_host = f"http://{api_host}:{api_port}"

        with mock.patch("openpilot.common.api.API_HOST", api_host), \
             mock.patch.dict("openpilot.selfdrive.test.helpers.managed_processes") as managed_processes_mock, \
             mock.patch("openpilot.system.updated.common.USERDATA", str(self.userdata)), \
             mock.patch("openpilot.system.updated.common.FINALIZED", str(self.finalized)):

          managed_processes_mock["updated"] = PythonProcess("updated", "system.updated.updated", only_offroad, enabled=not PC)

          with fake_ab("_a"):
            yield

  def setup_git_basedir_release(self, release):
    self.setup_basedir_release(release)
    run_cmd(["git", "init"], cwd=self.basedir)
    run_cmd(["git", "config", "user.name", "'tester'"], cwd=self.basedir)
    run_cmd(["git", "config", "user.email", "'tester@comma.ai'"], cwd=self.basedir)
    run_cmd(["git", "checkout", "-b", release], cwd=self.basedir)
    update_release(self.basedir, release, *self.MOCK_RELEASES[release])
    run_cmd(["git", "add", "."], cwd=self.basedir)
    run_cmd(["git", "commit", "-m", f"openpilot release {release}"], cwd=self.basedir)

  def _wait_for_finalized(self):
    self.wait_for_condition(lambda: get_valid_flag(self.finalized))
    time.sleep(1)

  def _test_channel_param(self, channel):
    self.assertEqual(self.params.get("UpdaterTargetChannel", encoding="utf-8"), channel)

  def test_no_update(self):
    # Start on release3, ensure we don't fetch any updates
    self.setup_remote_release("release3")
    self.setup_basedir_release("release3")

    with self.additional_context(), processes_context(["updated"]):
      time.sleep(1)
      self.assertEqual(get_valid_flag(self.finalized), False)

  def test_new_release(self):
    # Start on release3, simulate a release3 commit, ensure we fetch that update properly
    self.setup_remote_release("release3")
    self.setup_basedir_release("release3")

    self.MOCK_RELEASES["release3"] = ("0.1.3", "1.2", "0.1.3 release notes")
    self.update_remote_release("release3")

    with self.additional_context(), processes_context(["updated"]):
      time.sleep(1)
      self._wait_for_finalized()
      self._test_finalized_update("release3", *self.MOCK_RELEASES["release3"])

  def test_recover_from_git_update(self):
    # starts off on a git update, ensures we can recover and install the correct update
    self.setup_git_basedir_release("release3")
    self.setup_remote_release("release3")

    with self.additional_context(), processes_context(["updated"]):
      self._wait_for_finalized()
      self._test_finalized_update("release3", *self.MOCK_RELEASES["release3"])

  def test_channel_migration(self):
    # Start on 'test', remote has migrated this to 'release3', ensure we also switch to this branch
    self.MOCK_RELEASES["test"] = self.MOCK_RELEASES["release3"]
    self.setup_remote_release("release3")
    self.setup_remote_release("test")
    self.release_manifests["test"] = self.release_manifests["release3"]

    self.setup_git_basedir_release("test")
    self.setup_remote_release("release3")

    with self.additional_context(), processes_context(["updated"]):
      self._wait_for_finalized()
      self._test_finalized_update("release3", *self.MOCK_RELEASES["release3"])

  def test_recover_from_bad_api_response(self):
    # tests recovery from a bad api response
    self.setup_remote_release("release3")
    self.setup_basedir_release("release3")

    self.MOCK_RELEASES["release3"] = ("0.1.3", "1.2", "0.1.3 release notes")
    self.update_remote_release("release3")

    with self.additional_context():
      self.api_handler.API_NO_RESPONSE = True
      with processes_context(["updated"]):
        time.sleep(1)
        self.api_handler.API_NO_RESPONSE = False

        self._wait_for_finalized()
        self._test_finalized_update("release3", *self.MOCK_RELEASES["release3"])

  def test_recover_from_network_failure(self):
    # tests recovery from a network error on the directory
    self.setup_remote_release("release3")
    self.setup_basedir_release("release3")

    self.MOCK_RELEASES["release3"] = ("0.1.3", "1.2", "0.1.3 release notes")
    self.update_remote_release("release3")

    with self.additional_context():
      self.directory_handler.API_NO_RESPONSE = True
      with processes_context(["updated"]):
        time.sleep(1)
        self.directory_handler.API_NO_RESPONSE = False

        self._wait_for_finalized()
        self._test_finalized_update("release3", *self.MOCK_RELEASES["release3"])

  def test_agnos_update(self):
    self.setup_remote_release("release3")
    self.setup_basedir_release("release3")

    self.MOCK_RELEASES["release3"] = ("0.1.3", "1.3", "0.1.3 release notes")
    self.update_remote_release("release3")

    with self.additional_context(), \
      processes_context(["updated"]):

      time.sleep(1)
      self._wait_for_finalized()
      self._test_finalized_update("release3", *self.MOCK_RELEASES["release3"])

      # ensure update was pushed into correct slot
      self.assertEqual(self.boot_a.read_bytes()[:3], b"1.2")
      self.assertEqual(self.boot_b.read_bytes()[:3], b"1.3")

      self.assertEqual(self.system_a.read_bytes()[:3], b"1.2")
      self.assertEqual(self.system_b.read_bytes()[:3], b"1.3")

      self.assertEqual(len(self.system_b.read_bytes()), 3)
