#!/usr/bin/env python3

# Safe Update: A simple service that waits for network access and tries to
# update every 10 minutes. It's intended to make the OP update process more
# robust against Git repository corruption. This service DOES NOT try to fix
# an already-corrupt BASEDIR Git repo, only prevent it from happening.
#
# During normal operation, both onroad and offroad, the update process makes
# no changes to the BASEDIR install of OP. All update attempts are performed
# in a disposable staging area provided by OverlayFS. It assumes the deleter
# process provides enough disk space to carry out the process.
#
# If an update succeeds, a flag is set, and the update is swapped in at the
# next reboot. If an update is interrupted or otherwise fails, the OverlayFS
# upper layer and metadata can be discarded before trying again.
#
# The swap on boot is triggered by launch_chffrplus.sh
# gated on the existence of $FINALIZED/.overlay_consistent and also the
# existence and mtime of $BASEDIR/.overlay_init.
#
# Other than build byproducts, BASEDIR should not be modified while this
# service is running. Developers modifying code directly in BASEDIR should
# disable this service.

import os
import datetime
import subprocess
import psutil
import shutil
import signal
import fcntl
import time
import threading
from pathlib import Path
from typing import List, Tuple, Optional

from common.basedir import BASEDIR
from common.params import Params
from selfdrive.hardware import EON, TICI, HARDWARE
from selfdrive.swaglog import cloudlog
from selfdrive.controls.lib.alertmanager import set_offroad_alert

LOCK_FILE = os.getenv("UPDATER_LOCK_FILE", "/tmp/safe_staging_overlay.lock")
STAGING_ROOT = os.getenv("UPDATER_STAGING_ROOT", "/data/safe_staging")

NEOSUPDATE_DIR = os.getenv("UPDATER_NEOSUPDATE_DIR", "/data/neoupdate")

OVERLAY_UPPER = os.path.join(STAGING_ROOT, "upper")
OVERLAY_METADATA = os.path.join(STAGING_ROOT, "metadata")
OVERLAY_MERGED = os.path.join(STAGING_ROOT, "merged")
FINALIZED = os.path.join(STAGING_ROOT, "finalized")


class WaitTimeHelper:
  def __init__(self, proc):
    self.proc = proc
    self.ready_event = threading.Event()
    self.shutdown = False
    signal.signal(signal.SIGTERM, self.graceful_shutdown)
    signal.signal(signal.SIGINT, self.graceful_shutdown)
    signal.signal(signal.SIGHUP, self.update_now)

  def graceful_shutdown(self, signum: int, frame) -> None:
    # umount -f doesn't appear effective in avoiding "device busy" on NEOS,
    # so don't actually die until the next convenient opportunity in main().
    cloudlog.info("caught SIGINT/SIGTERM, dismounting overlay at next opportunity")

    # forward the signal to all our child processes
    child_procs = self.proc.children(recursive=True)
    for p in child_procs:
      p.send_signal(signum)

    self.shutdown = True
    self.ready_event.set()

  def update_now(self, signum: int, frame) -> None:
    cloudlog.info("caught SIGHUP, running update check immediately")
    self.ready_event.set()

  def sleep(self, t: float) -> None:
    self.ready_event.wait(timeout=t)


def run(cmd: List[str], cwd: Optional[str] = None, low_priority: bool = False):
  if low_priority:
    cmd = ["nice", "-n", "19"] + cmd
  return subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.STDOUT, encoding='utf8')


def set_consistent_flag(consistent: bool) -> None:
  os.sync()
  consistent_file = Path(os.path.join(FINALIZED, ".overlay_consistent"))
  if consistent:
    consistent_file.touch()
  elif not consistent:
    consistent_file.unlink(missing_ok=True)
  os.sync()


def set_params(new_version: bool, failed_count: int, exception: Optional[str]) -> None:
  params = Params()

  params.put("UpdateFailedCount", str(failed_count))
  if failed_count == 0:
    t = datetime.datetime.utcnow().isoformat()
    params.put("LastUpdateTime", t.encode('utf8'))

  if exception is None:
    params.delete("LastUpdateException")
  else:
    params.put("LastUpdateException", exception)

  if new_version:
    try:
      with open(os.path.join(FINALIZED, "RELEASES.md"), "rb") as f:
        r = f.read()
      r = r[:r.find(b'\n\n')]  # Slice latest release notes
      params.put("ReleaseNotes", r + b"\n")
    except Exception:
      params.put("ReleaseNotes", "")
    params.put_bool("UpdateAvailable", True)


def setup_git_options(cwd: str) -> None:
  # We sync FS object atimes (which NEOS doesn't use) and mtimes, but ctimes
  # are outside user control. Make sure Git is set up to ignore system ctimes,
  # because they change when we make hard links during finalize. Otherwise,
  # there is a lot of unnecessary churn. This appears to be a common need on
  # OSX as well: https://www.git-tower.com/blog/make-git-rebase-safe-on-osx/

  # We are using copytree to copy the directory, which also changes
  # inode numbers. Ignore those changes too.
  git_cfg = [
    ("core.trustctime", "false"),
    ("core.checkStat", "minimal"),
  ]
  for option, value in git_cfg:
    run(["git", "config", option, value], cwd)


def dismount_overlay() -> None:
  if os.path.ismount(OVERLAY_MERGED):
    cloudlog.info("unmounting existing overlay")
    args = ["umount", "-l", OVERLAY_MERGED]
    if TICI:
      args = ["sudo"] + args
    run(args)


def init_overlay() -> None:

  overlay_init_file = Path(os.path.join(BASEDIR, ".overlay_init"))

  # Re-create the overlay if BASEDIR/.git has changed since we created the overlay
  if overlay_init_file.is_file():
    git_dir_path = os.path.join(BASEDIR, ".git")
    new_files = run(["find", git_dir_path, "-newer", str(overlay_init_file)])
    if not len(new_files.splitlines()):
      # A valid overlay already exists
      return
    else:
      cloudlog.info(".git directory changed, recreating overlay")

  cloudlog.info("preparing new safe staging area")

  params = Params()
  params.put_bool("UpdateAvailable", False)
  set_consistent_flag(False)
  dismount_overlay()
  if TICI:
    run(["sudo", "rm", "-rf", STAGING_ROOT])
  if os.path.isdir(STAGING_ROOT):
    shutil.rmtree(STAGING_ROOT)

  for dirname in [STAGING_ROOT, OVERLAY_UPPER, OVERLAY_METADATA, OVERLAY_MERGED]:
    os.mkdir(dirname, 0o755)

  if os.lstat(BASEDIR).st_dev != os.lstat(OVERLAY_MERGED).st_dev:
    raise RuntimeError("base and overlay merge directories are on different filesystems; not valid for overlay FS!")

  # Leave a timestamped canary in BASEDIR to check at startup. The device clock
  # should be correct by the time we get here. If the init file disappears, or
  # critical mtimes in BASEDIR are newer than .overlay_init, continue.sh can
  # assume that BASEDIR has used for local development or otherwise modified,
  # and skips the update activation attempt.
  consistent_file = Path(os.path.join(BASEDIR, ".overlay_consistent"))
  if consistent_file.is_file():
    consistent_file.unlink()
  overlay_init_file.touch()

  os.sync()
  overlay_opts = f"lowerdir={BASEDIR},upperdir={OVERLAY_UPPER},workdir={OVERLAY_METADATA}"

  mount_cmd = ["mount", "-t", "overlay", "-o", overlay_opts, "none", OVERLAY_MERGED]
  if TICI:
    run(["sudo"] + mount_cmd)
    run(["sudo", "chmod", "755", os.path.join(OVERLAY_METADATA, "work")])
  else:
    run(mount_cmd)

  git_diff = run(["git", "diff"], OVERLAY_MERGED, low_priority=True)
  params.put("GitDiff", git_diff)
  cloudlog.info(f"git diff output:\n{git_diff}")


def finalize_update() -> None:
  """Take the current OverlayFS merged view and finalize a copy outside of
  OverlayFS, ready to be swapped-in at BASEDIR. Copy using shutil.copytree"""

  # Remove the update ready flag and any old updates
  cloudlog.info("creating finalized version of the overlay")
  set_consistent_flag(False)

  # Copy the merged overlay view and set the update ready flag
  if os.path.exists(FINALIZED):
    shutil.rmtree(FINALIZED)
  shutil.copytree(OVERLAY_MERGED, FINALIZED, symlinks=True)

  run(["git", "reset", "--hard"], FINALIZED)
  run(["git", "submodule", "foreach", "--recursive", "git", "reset"], FINALIZED)

  set_consistent_flag(True)
  cloudlog.info("done finalizing overlay")


def handle_agnos_update(wait_helper):
  from selfdrive.hardware.tici.agnos import flash_agnos_update, get_target_slot_number

  cur_version = HARDWARE.get_os_version()
  updated_version = run(["bash", "-c", r"unset AGNOS_VERSION && source launch_env.sh && \
                          echo -n $AGNOS_VERSION"], OVERLAY_MERGED).strip()

  cloudlog.info(f"AGNOS version check: {cur_version} vs {updated_version}")
  if cur_version == updated_version:
    return

  # prevent an openpilot getting swapped in with a mismatched or partially downloaded agnos
  set_consistent_flag(False)

  cloudlog.info(f"Beginning background installation for AGNOS {updated_version}")
  set_offroad_alert("Offroad_NeosUpdate", True)

  manifest_path = os.path.join(OVERLAY_MERGED, "selfdrive/hardware/tici/agnos.json")
  target_slot_number = get_target_slot_number()
  flash_agnos_update(manifest_path, target_slot_number, cloudlog)
  set_offroad_alert("Offroad_NeosUpdate", False)


def handle_neos_update(wait_helper: WaitTimeHelper) -> None:
  from selfdrive.hardware.eon.neos import download_neos_update

  cur_neos = HARDWARE.get_os_version()
  updated_neos = run(["bash", "-c", r"unset REQUIRED_NEOS_VERSION && source launch_env.sh && \
                       echo -n $REQUIRED_NEOS_VERSION"], OVERLAY_MERGED).strip()

  cloudlog.info(f"NEOS version check: {cur_neos} vs {updated_neos}")
  if cur_neos == updated_neos:
    return

  cloudlog.info(f"Beginning background download for NEOS {updated_neos}")
  set_offroad_alert("Offroad_NeosUpdate", True)

  update_manifest = os.path.join(OVERLAY_MERGED, "selfdrive/hardware/eon/neos.json")

  neos_downloaded = False
  start_time = time.monotonic()
  # Try to download for one day
  while not neos_downloaded and not wait_helper.shutdown and \
        (time.monotonic() - start_time < 60*60*24):
    wait_helper.ready_event.clear()
    try:
      download_neos_update(update_manifest, cloudlog)
      neos_downloaded = True
    except Exception:
      cloudlog.info("NEOS background download failed, retrying")
      wait_helper.sleep(120)

  # If the download failed, we'll show the alert again when we retry
  set_offroad_alert("Offroad_NeosUpdate", False)
  if not neos_downloaded:
    raise Exception("Failed to download NEOS update")
  cloudlog.info(f"NEOS background download successful, took {time.monotonic() - start_time} seconds")


def check_git_fetch_result(fetch_txt):
  err_msg = "Failed to add the host to the list of known hosts (/data/data/com.termux/files/home/.ssh/known_hosts).\n"
  return len(fetch_txt) > 0 and (fetch_txt != err_msg)


def check_for_update() -> Tuple[bool, bool]:
  setup_git_options(OVERLAY_MERGED)
  try:
    git_fetch_output = run(["git", "fetch", "--dry-run"], OVERLAY_MERGED, low_priority=True)
    return True, check_git_fetch_result(git_fetch_output)
  except subprocess.CalledProcessError:
    return False, False


def fetch_update(wait_helper: WaitTimeHelper) -> bool:
  cloudlog.info("attempting git fetch inside staging overlay")

  setup_git_options(OVERLAY_MERGED)

  git_fetch_output = run(["git", "fetch"], OVERLAY_MERGED, low_priority=True)
  cloudlog.info("git fetch success: %s", git_fetch_output)

  cur_hash = run(["git", "rev-parse", "HEAD"], OVERLAY_MERGED).rstrip()
  upstream_hash = run(["git", "rev-parse", "@{u}"], OVERLAY_MERGED).rstrip()
  new_version = cur_hash != upstream_hash
  git_fetch_result = check_git_fetch_result(git_fetch_output)

  cloudlog.info(f"comparing {cur_hash} to {upstream_hash}")
  if new_version or git_fetch_result:
    cloudlog.info("Running update")

    if new_version:
      cloudlog.info("git reset in progress")
      r = [
        run(["git", "reset", "--hard", "@{u}"], OVERLAY_MERGED, low_priority=True),
        run(["git", "clean", "-xdf"], OVERLAY_MERGED, low_priority=True ),
        run(["git", "submodule", "init"], OVERLAY_MERGED, low_priority=True),
        run(["git", "submodule", "update"], OVERLAY_MERGED, low_priority=True),
      ]
      cloudlog.info("git reset success: %s", '\n'.join(r))

      if EON:
        handle_neos_update(wait_helper)
      elif TICI:
        handle_agnos_update(wait_helper)

    # Create the finalized, ready-to-swap update
    finalize_update()
    cloudlog.info("openpilot update successful!")
  else:
    cloudlog.info("nothing new from git at this time")

  return new_version


def main():
  params = Params()

  if params.get_bool("DisableUpdates"):
    raise RuntimeError("updates are disabled by the DisableUpdates param")

  if EON and os.geteuid() != 0:
    raise RuntimeError("updated must be launched as root!")

  ov_lock_fd = open(LOCK_FILE, 'w')
  try:
    fcntl.flock(ov_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
  except IOError as e:
    raise RuntimeError("couldn't get overlay lock; is another updated running?") from e

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

  # Wait for IsOffroad to be set before our first update attempt
  wait_helper = WaitTimeHelper(proc)
  wait_helper.sleep(30)

  overlay_init = Path(os.path.join(BASEDIR, ".overlay_init"))
  overlay_init.unlink(missing_ok=True)

  first_run = True
  last_fetch_time = 0
  update_failed_count = 0

  # Run the update loop
  #  * every 1m, do a lightweight internet/update check
  #  * every 10m, do a full git fetch
  while not wait_helper.shutdown:
    update_now = wait_helper.ready_event.is_set()
    wait_helper.ready_event.clear()

    # Don't run updater while onroad or if the time's wrong
    time_wrong = datetime.datetime.utcnow().year < 2019
    is_onroad = not params.get_bool("IsOffroad")
    if is_onroad or time_wrong:
      wait_helper.sleep(30)
      cloudlog.info("not running updater, not offroad")
      continue

    # Attempt an update
    exception = None
    new_version = False
    update_failed_count += 1
    try:
      init_overlay()

      internet_ok, update_available = check_for_update()
      if internet_ok and not update_available:
        update_failed_count = 0

      # Fetch updates at most every 10 minutes
      if internet_ok and (update_now or time.monotonic() - last_fetch_time > 60*10):
        new_version = fetch_update(wait_helper)
        update_failed_count = 0
        last_fetch_time = time.monotonic()

        if first_run and not new_version and os.path.isdir(NEOSUPDATE_DIR):
          shutil.rmtree(NEOSUPDATE_DIR)
        first_run = False
    except subprocess.CalledProcessError as e:
      cloudlog.event(
        "update process failed",
        cmd=e.cmd,
        output=e.output,
        returncode=e.returncode
      )
      exception = f"command failed: {e.cmd}\n{e.output}"
      overlay_init.unlink(missing_ok=True)
    except Exception as e:
      cloudlog.exception("uncaught updated exception, shouldn't happen")
      exception = str(e)
      overlay_init.unlink(missing_ok=True)

    set_params(new_version, update_failed_count, exception)
    wait_helper.sleep(60)

  dismount_overlay()


if __name__ == "__main__":
  main()
