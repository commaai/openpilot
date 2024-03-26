from enum import StrEnum
import os
from pathlib import Path
import shutil
from markdown_it import MarkdownIt
import requests

from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.updated.tests.test_base import get_consistent_flag
from openpilot.selfdrive.updated.updated import UserRequest, WaitTimeHelper, handle_agnos_update
from openpilot.system.hardware import AGNOS
from openpilot.system.updated.casync.read import extract_remote
from openpilot.system.version import BuildMetadata, get_build_metadata, build_metadata_from_dict

UPDATE_DELAY = 60
CHANNELS_API_ROOT = "v1/openpilot/channels"

Manifest = dict

API_HOST = os.getenv('API_HOST', 'https://api.commadotai.com')

def get_available_channels() -> list | None:
  try:
    return list(requests.get(f"{API_HOST}/{CHANNELS_API_ROOT}").json())
  except Exception:
    cloudlog.exception("fetching remote channels")
    return None

def get_remote_channel_data(channel) -> tuple[BuildMetadata | None, Manifest | None]:
  try:
    data = requests.get(f"{API_HOST}/{CHANNELS_API_ROOT}/{channel}").json()
    return build_metadata_from_dict(data["build_metadata"]), data["manifest"]
  except Exception:
    cloudlog.exception("fetching remote manifest failed")
    return None, None


LOCK_FILE = os.getenv("UPDATER_LOCK_FILE", "/tmp/safe_staging_overlay.lock")
STAGING_ROOT = os.getenv("UPDATER_STAGING_ROOT", "/data/safe_staging")

CASYNC_PATH = Path(STAGING_ROOT) / "casync"        # where the casync update is pulled
CASYNC_TMPDIR = Path(STAGING_ROOT) / "casync_tmp"  # working directory for casync temp files
FINALIZED = os.path.join(STAGING_ROOT, "finalized")


def set_consistent_flag(consistent: bool) -> None:
  os.sync()
  consistent_file = Path(os.path.join(FINALIZED, ".overlay_consistent"))
  if consistent:
    consistent_file.touch()
  elif not consistent:
    consistent_file.unlink(missing_ok=True)
  os.sync()


class UpdaterState(StrEnum):
  IDLE = "idle"
  CHECKING = "checking..."
  DOWNLOADING = "downloading..."
  FINALIZING = "finalizing update..."
  FAILED = "failed to check for update..."


def set_status_params(state: UpdaterState, update_available = False, update_ready = False):
  params = Params()
  params.put("UpdaterState", state)
  params.put_bool("UpdaterFetchAvailable", update_available)
  params.put_bool("UpdateAvailable", update_ready)


def set_channel_params(name, build_metadata: BuildMetadata):
  params = Params()
  params.put(f"Updater{name}Description", f'{build_metadata.openpilot.version} / {build_metadata.channel}')
  params.put(f"Updater{name}ReleaseNotes", bytes(MarkdownIt().render(build_metadata.openpilot.release_notes), encoding="utf-8"))


def set_current_channel_params(build_metadata: BuildMetadata):
  set_channel_params("Current", build_metadata)


def set_new_channel_params(build_metadata: BuildMetadata):
  set_channel_params("New", build_metadata)



def check_update_available(current_directory, other_metadata: BuildMetadata):
  build_metadata = get_build_metadata(current_directory)
  return build_metadata.channel != build_metadata.channel or \
         build_metadata.openpilot.git_commit != other_metadata.openpilot.git_commit


def download_update(manifest):
  cloudlog.info("")
  env = os.environ.copy()
  env["TMPDIR"] = str(CASYNC_TMPDIR)
  CASYNC_TMPDIR.mkdir(exist_ok=True)
  CASYNC_PATH.mkdir(exist_ok=True)

  for entry in manifest:
    if "type" in entry and entry["type"] == "path" and entry["path"] == "/data/openpilot":
      extract_remote(entry["casync"]["caidx"], CASYNC_PATH)


def finalize_update():
  cloudlog.info("creating finalized version of the overlay")
  set_consistent_flag(False)

  if os.path.exists(FINALIZED):
    shutil.rmtree(FINALIZED)
  shutil.copytree(CASYNC_PATH, FINALIZED, symlinks=True)

  set_consistent_flag(True)
  cloudlog.info("done finalizing overlay")


def main():
  params = Params()
  set_status_params(UpdaterState.CHECKING)

  wait_helper = WaitTimeHelper()

  while True:
    wait_helper.ready_event.clear()

    target_channel = params.get("UpdaterTargetBranch", encoding='utf8')
    build_metadata = get_build_metadata()

    if target_channel is None:
      target_channel = build_metadata.channel
      params.put("UpdaterTargetBranch", target_channel)

    user_requested_check = wait_helper.user_request == UserRequest.CHECK

    set_status_params(UpdaterState.CHECKING)

    update_ready = get_consistent_flag(FINALIZED)

    set_current_channel_params(build_metadata)

    remote_build_metadata, remote_manifest = get_remote_channel_data(target_channel)

    if remote_build_metadata is not None or remote_manifest is not None:
      update_available = check_update_available(BASEDIR, remote_build_metadata)

      if update_ready and not check_update_available(FINALIZED, remote_build_metadata):
        update_available = False

      set_status_params(UpdaterState.IDLE, update_available, update_ready)

      if update_available:
        if user_requested_check:
          cloudlog.info("skipping fetch, only checking")
        else:
          update_available = False
          set_status_params(UpdaterState.DOWNLOADING)
          try:
            download_update(remote_manifest)
            if AGNOS:
              handle_agnos_update(CASYNC_PATH)

            set_status_params(UpdaterState.FINALIZING)
            finalize_update()
            new_build_metadata = get_build_metadata(FINALIZED)
            set_new_channel_params(new_build_metadata)
            update_ready = get_consistent_flag(FINALIZED)
            set_status_params(UpdaterState.IDLE, update_available, update_ready)
          except Exception:
            set_status_params(UpdaterState.FAILED, False, False)
            cloudlog.exception("exception while downloading ...")
    else:
      set_status_params(UpdaterState.FAILED, False, False)

    wait_helper.user_request = UserRequest.NONE
    wait_helper.sleep(UPDATE_DELAY)


if __name__ == "__main__":
  main()
