import contextlib
import dataclasses
import http
import json
import os
import pathlib

from openpilot.selfdrive.test.helpers import DirectoryHttpServer, http_server_context, processes_context
from openpilot.system.updated.casync.common import create_build_metadata_file, create_caexclude_file, create_casync_release
from openpilot.system.version import BuildMetadata, OpenpilotMetadata
from openpilot.selfdrive.updated.tests.test_base import BaseUpdateTest, run, update_release


def create_remote_response(channel, build_metadata, casync_caidx, casync_digest):
  return {
    "build_metadata": dataclasses.asdict(build_metadata),
    "manifest": [
      {
        "type": "path",
        "path": "/data/openpilot",
        "casync": {
          "caidx": casync_caidx,
          "digest": casync_digest
        }
      }
    ]
  }


def create_mock_build_metadata(channel, version, agnos_version, release_notes):
  commit = hash((version, agnos_version, release_notes))  # simulated commit
  return BuildMetadata(channel, OpenpilotMetadata(version, release_notes, commit, "https://github.com/commaai/openpilot.git", "2024", "release", False))


def create_casync_files(dirname, channel, version, agnos_version, release_notes):
  create_caexclude_file(pathlib.Path(dirname))
  create_build_metadata_file(pathlib.Path(dirname), create_mock_build_metadata(channel, version, agnos_version, release_notes), channel)


def OpenpilotChannelMockAPI(release_digests, mock_releases, casync_base):
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
        response = list(release_digests.keys())
      else:
        channel = self.path.split("/")[-1]
        build_metadata = create_mock_build_metadata(channel, *mock_releases[channel])
        response = create_remote_response(channel, build_metadata, f"{casync_base}/{channel}.caidx", release_digests[channel])

      response = json.dumps(response)

      self.send_response(200)
      self.send_header('Content-Type', 'application/json')
      self.end_headers()
      self.wfile.write(response.encode(encoding='utf_8'))

  return Handler


class TestUpdateDCASyncStrategy(BaseUpdateTest):
  def setUp(self):
    super().setUp()
    self.casync_dir = self.mock_update_path / "casync"
    self.casync_dir.mkdir()
    self.release_digests = {}

  def update_remote_release(self, release):
    update_release(self.remote_dir, release, *self.MOCK_RELEASES[release])
    create_casync_files(self.remote_dir, release, *self.MOCK_RELEASES[release])
    self.release_digests[release] = create_casync_release(self.remote_dir, self.casync_dir, release)[0]

  def setup_remote_release(self, release):
    self.update_remote_release(release)

  def setup_basedir_release(self, release):
    super().setup_basedir_release(release)
    update_release(self.basedir, release, *self.MOCK_RELEASES[release])
    create_casync_files(self.basedir, release, *self.MOCK_RELEASES[release])

  @contextlib.contextmanager
  def additional_context(self):
    self.directory_handler = DirectoryHttpServer(self.casync_dir)
    with http_server_context(self.directory_handler) as (casync_host, casync_port):
      casync_base = f"http://{casync_host}:{casync_port}"

      self.api_handler = OpenpilotChannelMockAPI(self.release_digests, self.MOCK_RELEASES, casync_base)
      with http_server_context(self.api_handler) as (api_host, api_port):
        os.environ["API_HOST"] = f"http://{api_host}:{api_port}"
        yield

  def setup_git_basedir_release(self, release):
    super().setup_basedir_release(release)
    run(["git", "init"], cwd=self.basedir)
    run(["git", "checkout", "-b", release], cwd=self.basedir)
    update_release(self.basedir, release, *self.MOCK_RELEASES[release])
    run(["git", "add", "."], cwd=self.basedir)
    run(["git", "commit", "-m", f"openpilot release {release}"], cwd=self.basedir)

  def test_recover_from_git_update(self):
    # starts off on a git update, ensures we can recover and install the correct update
    self.setup_git_basedir_release("release3")
    self.setup_remote_release("release3")

    with self.additional_context(), processes_context(["updated"]) as [_]:
      self._test_params("release3", False, False)
      self.wait_for_idle()
      self._test_params("release3", False, True)

      self._test_finalized_update("release3", *self.MOCK_RELEASES["release3"])

  def test_recover_from_bad_api_response(self):
    # Start on release3, ensure we don't fetch any updates
    self.setup_remote_release("release3")
    self.setup_basedir_release("release3")

    with self.additional_context():
      self.api_handler.API_NO_RESPONSE = True
      with processes_context(["updated"]) as [updated]:
        self._test_params("release3", False, False)
        self.wait_for_failed()
        self._test_params("release3", False, False)

        self.send_check_for_updates_signal(updated)

        self.wait_for_failed()
        self._test_params("release3", False, False)

        self.api_handler.API_NO_RESPONSE = False

        self.send_check_for_updates_signal(updated)
        self.wait_for_idle()
        self._test_params("release3", False, False)

  def test_recover_from_network_failure(self):
    # Start on release3, ensure we don't fetch any updates
    self.setup_remote_release("release3")
    self.setup_basedir_release("release3")

    with self.additional_context():
      self.directory_handler.API_NO_RESPONSE = True
      with processes_context(["updated"]) as [updated]:
        self._test_params("release3", False, False)
        self.wait_for_idle()
        self._test_params("release3", False, False)

        self.MOCK_RELEASES["release3"] = ("0.1.3", "1.2", "0.1.3 release notes")
        self.update_remote_release("release3")

        self.send_check_for_updates_signal(updated)
        self.wait_for_fetch_available()

        self._test_params("release3", True, False)

        self.send_download_signal(updated)

        self.wait_for_failed()

        self.directory_handler.API_NO_RESPONSE = False

        self.send_download_signal(updated)

        self.wait_for_update_available()

        self._test_params("release3", False, True)
        self._test_finalized_update("release3", *self.MOCK_RELEASES["release3"])
