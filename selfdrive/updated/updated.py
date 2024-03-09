#!/usr/bin/env python3
import os
from pathlib import Path
import datetime
import subprocess
import psutil
import signal
import fcntl
import threading

from openpilot.common.params import Params
from openpilot.common.time import system_time_valid
from openpilot.selfdrive.updated.common import LOCK_FILE, STAGING_ROOT, UpdateStrategy, run, set_consistent_flag
from openpilot.system.hardware import AGNOS, HARDWARE
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.controls.lib.alertmanager import set_offroad_alert
from openpilot.system.version import is_tested_branch
from openpilot.selfdrive.updated.git import GitUpdateStrategy

DAYS_NO_CONNECTIVITY_MAX = 14     # do not allow to engage after this many days
DAYS_NO_CONNECTIVITY_PROMPT = 10  # send an offroad prompt after this many days

class UserRequest:
  NONE = 0
  CHECK = 1
  FETCH = 2

class WaitTimeHelper:
  def __init__(self):
    self.ready_event = threading.Event()
    self.user_request = UserRequest.NONE
    signal.signal(signal.SIGHUP, self.update_now)
    signal.signal(signal.SIGUSR1, self.check_now)

  def update_now(self, signum: int, frame) -> None:
    cloudlog.info("caught SIGHUP, attempting to downloading update")
    self.user_request = UserRequest.FETCH
    self.ready_event.set()

  def check_now(self, signum: int, frame) -> None:
    cloudlog.info("caught SIGUSR1, checking for updates")
    self.user_request = UserRequest.CHECK
    self.ready_event.set()

  def sleep(self, t: float) -> None:
    self.ready_event.wait(timeout=t)

def write_time_to_param(params, param) -> None:
  t = datetime.datetime.utcnow()
  params.put(param, t.isoformat().encode('utf8'))

def read_time_from_param(params, param) -> datetime.datetime | None:
  t = params.get(param, encoding='utf8')
  try:
    return datetime.datetime.fromisoformat(t)
  except (TypeError, ValueError):
    pass
  return None


def handle_agnos_update(fetched_path) -> None:
  from openpilot.system.hardware.tici.agnos import flash_agnos_update, get_target_slot_number

  cur_version = HARDWARE.get_os_version()
  updated_version = run(["bash", "-c", r"unset AGNOS_VERSION && source launch_env.sh && \
                          echo -n $AGNOS_VERSION"], fetched_path).strip()

  cloudlog.info(f"AGNOS version check: {cur_version} vs {updated_version}")
  if cur_version == updated_version:
    return

  # prevent an openpilot getting swapped in with a mismatched or partially downloaded agnos
  set_consistent_flag(False)

  cloudlog.info(f"Beginning background installation for AGNOS {updated_version}")
  set_offroad_alert("Offroad_NeosUpdate", True)

  manifest_path = os.path.join(fetched_path, "system/hardware/tici/agnos.json")
  target_slot_number = get_target_slot_number()
  flash_agnos_update(manifest_path, target_slot_number, cloudlog)
  set_offroad_alert("Offroad_NeosUpdate", False)


STRATEGY = {
  "git": GitUpdateStrategy,
}


class Updater:
  def __init__(self):
    self.params = Params()
    self._has_internet: bool = False

    self.strategy: UpdateStrategy = STRATEGY[os.environ.get("UPDATER_STRATEGY", "git")]()

  @property
  def has_internet(self) -> bool:
    return self._has_internet

  def init(self):
    self.strategy.init()

  def cleanup(self):
    self.strategy.cleanup()

  def set_params(self, update_success: bool, failed_count: int, exception: str | None) -> None:
    self.params.put("UpdateFailedCount", str(failed_count))

    if self.params.get("UpdaterTargetBranch") is None:
      self.params.put("UpdaterTargetBranch", self.strategy.current_channel())

    self.params.put_bool("UpdaterFetchAvailable", self.strategy.update_available())

    available_channels = self.strategy.get_available_channels()
    self.params.put("UpdaterAvailableBranches", ','.join(available_channels))

    last_update = datetime.datetime.utcnow()
    if update_success:
      write_time_to_param(self.params, "LastUpdateTime")
    else:
      t = read_time_from_param(self.params, "LastUpdateTime")
      if t is not None:
        last_update = t

    if exception is None:
      self.params.remove("LastUpdateException")
    else:
      self.params.put("LastUpdateException", exception)

    description_current, release_notes_current = self.strategy.describe_current_channel()
    description_ready, release_notes_ready = self.strategy.describe_ready_channel()

    self.params.put("UpdaterCurrentDescription", description_current)
    self.params.put("UpdaterCurrentReleaseNotes", release_notes_current)
    self.params.put("UpdaterNewDescription", description_ready)
    self.params.put("UpdaterNewReleaseNotes", release_notes_ready)
    self.params.put_bool("UpdateAvailable", self.strategy.update_ready())

    # Handle user prompt
    for alert in ("Offroad_UpdateFailed", "Offroad_ConnectivityNeeded", "Offroad_ConnectivityNeededPrompt"):
      set_offroad_alert(alert, False)

    now = datetime.datetime.utcnow()
    dt = now - last_update
    if failed_count > 15 and exception is not None and self.has_internet:
      if is_tested_branch():
        extra_text = "Ensure the software is correctly installed. Uninstall and re-install if this error persists."
      else:
        extra_text = exception
      set_offroad_alert("Offroad_UpdateFailed", True, extra_text=extra_text)
    elif failed_count > 0:
      if dt.days > DAYS_NO_CONNECTIVITY_MAX:
        set_offroad_alert("Offroad_ConnectivityNeeded", True)
      elif dt.days > DAYS_NO_CONNECTIVITY_PROMPT:
        remaining = max(DAYS_NO_CONNECTIVITY_MAX - dt.days, 1)
        set_offroad_alert("Offroad_ConnectivityNeededPrompt", True, extra_text=f"{remaining} day{'' if remaining == 1 else 's'}.")

  def check_for_update(self) -> None:
    cloudlog.info("checking for updates")

    self.strategy.update_available()

  def fetch_update(self) -> None:
    self.params.put("UpdaterState", "downloading...")

    # TODO: cleanly interrupt this and invalidate old update
    set_consistent_flag(False)
    self.params.put_bool("UpdateAvailable", False)

    self.strategy.fetch_update()

    # TODO: show agnos download progress
    if AGNOS:
      handle_agnos_update(self.strategy.fetched_path())

    # Create the finalized, ready-to-swap update
    self.params.put("UpdaterState", "finalizing update...")
    self.strategy.finalize_update()
    cloudlog.info("finalize success!")


def main() -> None:
  params = Params()

  if params.get_bool("DisableUpdates"):
    cloudlog.warning("updates are disabled by the DisableUpdates param")
    exit(0)

  with open(LOCK_FILE, 'w') as ov_lock_fd:
    try:
      fcntl.flock(ov_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
      raise RuntimeError("couldn't get overlay lock; is another instance running?") from e

    # Set low io priority
    proc = psutil.Process()
    if psutil.LINUX:
      proc.ionice(psutil.IOPRIO_CLASS_BE, value=7)

    # Check if we just performed an update
    if Path(os.path.join(STAGING_ROOT, "old_openpilot")).is_dir():
      cloudlog.event("update installed")

    if not params.get("InstallDate"):
      t = datetime.datetime.utcnow().isoformat()
      params.put("InstallDate", t.encode('utf8'))

    updater = Updater()
    update_failed_count = 0 # TODO: Load from param?
    wait_helper = WaitTimeHelper()

    # invalidate old finalized update
    set_consistent_flag(False)

    # set initial state
    params.put("UpdaterState", "idle")

    # Run the update loop
    first_run = True
    while True:
      wait_helper.ready_event.clear()

      # Attempt an update
      exception = None
      try:
        # TODO: reuse overlay from previous updated instance if it looks clean
        updater.init()

        # ensure we have some params written soon after startup
        updater.set_params(False, update_failed_count, exception)

        if not system_time_valid() or first_run:
          first_run = False
          wait_helper.sleep(60)
          continue

        update_failed_count += 1

        # check for update
        params.put("UpdaterState", "checking...")
        updater.check_for_update()

        # download update
        last_fetch = read_time_from_param(params, "UpdaterLastFetchTime")
        timed_out = last_fetch is None or (datetime.datetime.utcnow() - last_fetch > datetime.timedelta(days=3))
        user_requested_fetch = wait_helper.user_request == UserRequest.FETCH
        if params.get_bool("NetworkMetered") and not timed_out and not user_requested_fetch:
          cloudlog.info("skipping fetch, connection metered")
        elif wait_helper.user_request == UserRequest.CHECK:
          cloudlog.info("skipping fetch, only checking")
        else:
          updater.fetch_update()
          write_time_to_param(params, "UpdaterLastFetchTime")
        update_failed_count = 0
      except subprocess.CalledProcessError as e:
        cloudlog.event(
          "update process failed",
          cmd=e.cmd,
          output=e.output,
          returncode=e.returncode
        )
        exception = f"command failed: {e.cmd}\n{e.output}"
        updater.cleanup()
      except Exception as e:
        cloudlog.exception("uncaught updated exception, shouldn't happen")
        exception = str(e)
        updater.cleanup()

      try:
        params.put("UpdaterState", "idle")
        update_successful = (update_failed_count == 0)
        updater.set_params(update_successful, update_failed_count, exception)
      except Exception:
        cloudlog.exception("uncaught updated exception while setting params, shouldn't happen")

      # infrequent attempts if we successfully updated recently
      wait_helper.user_request = UserRequest.NONE
      wait_helper.sleep(5*60 if update_failed_count > 0 else 1.5*60*60)


if __name__ == "__main__":
  main()
