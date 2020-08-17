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

from common.basedir import BASEDIR
from common.params import Params
from selfdrive.swaglog import cloudlog
from selfdrive.controls.lib.alertmanager import set_offroad_alert

TEST_IP = os.getenv("UPDATER_TEST_IP", "8.8.8.8")
LOCK_FILE = os.getenv("UPDATER_LOCK_FILE", "/tmp/safe_staging_overlay.lock")
STAGING_ROOT = os.getenv("UPDATER_STAGING_ROOT", "/data/safe_staging")

NEOS_VERSION = os.getenv("UPDATER_NEOS_VERSION", "/VERSION")
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

  def graceful_shutdown(self, signum, frame):
    # umount -f doesn't appear effective in avoiding "device busy" on NEOS,
    # so don't actually die until the next convenient opportunity in main().
    cloudlog.info("caught SIGINT/SIGTERM, dismounting overlay at next opportunity")

    # forward the signal to all our child processes
    child_procs = self.proc.children(recursive=True)
    for p in child_procs:
      p.send_signal(signum)

    self.shutdown = True
    self.ready_event.set()

  def update_now(self, signum, frame):
    cloudlog.info("caught SIGHUP, running update check immediately")
    self.ready_event.set()

  def sleep(self, t):
    self.ready_event.wait(timeout=t)


def run(cmd, cwd=None, low_priority=False):
  if low_priority:
    cmd = ["nice", "-n", "19"] + cmd
  return subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.STDOUT, encoding='utf8')


def set_consistent_flag(consistent):
  os.system("sync")
  consistent_file = Path(os.path.join(FINALIZED, ".overlay_consistent"))
  if consistent:
    consistent_file.touch()
  elif not consistent and consistent_file.exists():
    consistent_file.unlink()
  os.system("sync")


def set_update_available_params(new_version):
  params = Params()

  t = datetime.datetime.utcnow().isoformat()
  params.put("LastUpdateTime", t.encode('utf8'))

  if new_version:
    try:
      with open(os.path.join(FINALIZED, "RELEASES.md"), "rb") as f:
        r = f.read()
      r = r[:r.find(b'\n\n')]  # Slice latest release notes
      params.put("ReleaseNotes", r + b"\n")
    except Exception:
      params.put("ReleaseNotes", "")
    params.put("UpdateAvailable", "1")


def dismount_ovfs():
  if os.path.ismount(OVERLAY_MERGED):
    cloudlog.error("unmounting existing overlay")
    run(["umount", "-l", OVERLAY_MERGED])


def setup_git_options(cwd):
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
    try:
      ret = run(["git", "config", "--get", option], cwd)
      config_ok = ret.strip() == value
    except subprocess.CalledProcessError:
      config_ok = False

    if not config_ok:
      cloudlog.info(f"Setting git '{option}' to '{value}'")
      run(["git", "config", option, value], cwd)


def init_ovfs():
  cloudlog.info("preparing new safe staging area")
  Params().put("UpdateAvailable", "0")

  set_consistent_flag(False)

  dismount_ovfs()
  if os.path.isdir(STAGING_ROOT):
    shutil.rmtree(STAGING_ROOT)

  for dirname in [STAGING_ROOT, OVERLAY_UPPER, OVERLAY_METADATA, OVERLAY_MERGED, FINALIZED]:
    os.mkdir(dirname, 0o755)

  if not os.lstat(BASEDIR).st_dev == os.lstat(OVERLAY_MERGED).st_dev:
    raise RuntimeError("base and overlay merge directories are on different filesystems; not valid for overlay FS!")

  # Remove consistent flag from current BASEDIR so it's not copied over
  if os.path.isfile(os.path.join(BASEDIR, ".overlay_consistent")):
    os.remove(os.path.join(BASEDIR, ".overlay_consistent"))

  # Leave a timestamped canary in BASEDIR to check at startup. The device clock
  # should be correct by the time we get here. If the init file disappears, or
  # critical mtimes in BASEDIR are newer than .overlay_init, continue.sh can
  # assume that BASEDIR has used for local development or otherwise modified,
  # and skips the update activation attempt.
  Path(os.path.join(BASEDIR, ".overlay_init")).touch()

  os.system("sync")
  overlay_opts = f"lowerdir={BASEDIR},upperdir={OVERLAY_UPPER},workdir={OVERLAY_METADATA}"
  run(["mount", "-t", "overlay", "-o", overlay_opts, "none", OVERLAY_MERGED])


def finalize_from_ovfs():
  """Take the current OverlayFS merged view and finalize a copy outside of
  OverlayFS, ready to be swapped-in at BASEDIR. Copy using shutil.copytree"""

  # Remove the update ready flag and any old updates
  cloudlog.info("creating finalized version of the overlay")
  set_consistent_flag(False)
  shutil.rmtree(FINALIZED)

  # Copy the merged overlay view and set the update ready flag
  shutil.copytree(OVERLAY_MERGED, FINALIZED, symlinks=True)
  set_consistent_flag(True)
  cloudlog.info("done finalizing overlay")


def attempt_update(wait_helper):
  cloudlog.info("attempting git update inside staging overlay")

  setup_git_options(OVERLAY_MERGED)

  git_fetch_output = run(["git", "fetch"], OVERLAY_MERGED, low_priority=True)
  cloudlog.info("git fetch success: %s", git_fetch_output)

  cur_hash = run(["git", "rev-parse", "HEAD"], OVERLAY_MERGED).rstrip()
  upstream_hash = run(["git", "rev-parse", "@{u}"], OVERLAY_MERGED).rstrip()
  new_version = cur_hash != upstream_hash

  err_msg = "Failed to add the host to the list of known hosts (/data/data/com.termux/files/home/.ssh/known_hosts).\n"
  git_fetch_result = len(git_fetch_output) > 0 and (git_fetch_output != err_msg)

  cloudlog.info("comparing %s to %s" % (cur_hash, upstream_hash))
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

      # Download the accompanying NEOS version if it doesn't match the current version
      with open(NEOS_VERSION, "r") as f:
        cur_neos = f.read().strip()

      updated_neos = run(["bash", "-c", r"unset REQUIRED_NEOS_VERSION && source launch_env.sh && \
                           echo -n $REQUIRED_NEOS_VERSION"], OVERLAY_MERGED).strip()

      cloudlog.info(f"NEOS version check: {cur_neos} vs {updated_neos}")
      if cur_neos != updated_neos:
        cloudlog.info(f"Beginning background download for NEOS {updated_neos}")

        set_offroad_alert("Offroad_NeosUpdate", True)
        updater_path = os.path.join(OVERLAY_MERGED, "installer/updater/updater")
        update_manifest = f"file://{OVERLAY_MERGED}/installer/updater/update.json"

        neos_downloaded = False
        start_time = time.monotonic()
        # Try to download for one day
        while (time.monotonic() - start_time < 60*60*24) and not wait_helper.shutdown:
          wait_helper.ready_event.clear()
          try:
            run([updater_path, "bgcache", update_manifest], OVERLAY_MERGED, low_priority=True)
            neos_downloaded = True
            break
          except subprocess.CalledProcessError:
            cloudlog.info("NEOS background download failed, retrying")
            wait_helper.sleep(120)

        # If the download failed, we'll show the alert again when we retry
        set_offroad_alert("Offroad_NeosUpdate", False)
        if not neos_downloaded:
          raise Exception("Failed to download NEOS update")

        cloudlog.info(f"NEOS background download successful, took {time.monotonic() - start_time} seconds")

    # Create the finalized, ready-to-swap update
    finalize_from_ovfs()
    cloudlog.info("openpilot update successful!")
  else:
    cloudlog.info("nothing new from git at this time")

  set_update_available_params(new_version)
  return new_version


def main():
  params = Params()

  if params.get("DisableUpdates") == b"1":
    raise RuntimeError("updates are disabled by the DisableUpdates param")

  if os.geteuid() != 0:
    raise RuntimeError("updated must be launched as root!")

  # Set low io priority
  proc = psutil.Process()
  if psutil.LINUX:
    proc.ionice(psutil.IOPRIO_CLASS_BE, value=7)

  ov_lock_fd = open(LOCK_FILE, 'w')
  try:
    fcntl.flock(ov_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
  except IOError:
    raise RuntimeError("couldn't get overlay lock; is another updated running?")

  # Wait for IsOffroad to be set before our first update attempt
  wait_helper = WaitTimeHelper(proc)
  wait_helper.sleep(30)

  update_failed_count = 0
  update_available = False
  overlay_initialized = False
  while not wait_helper.shutdown:
    wait_helper.ready_event.clear()

    # Check for internet every 30s
    time_wrong = datetime.datetime.utcnow().year < 2019
    ping_failed = os.system(f"ping -W 4 -c 1 {TEST_IP}") != 0
    if ping_failed or time_wrong:
      wait_helper.sleep(30)
      continue

    # Attempt an update
    exception = None
    update_failed_count += 1
    try:
      # Re-create the overlay if BASEDIR/.git has changed since we created the overlay
      if overlay_initialized:
        overlay_init_fn = os.path.join(BASEDIR, ".overlay_init")
        git_dir_path = os.path.join(BASEDIR, ".git")
        new_files = run(["find", git_dir_path, "-newer", overlay_init_fn])

        if len(new_files.splitlines()):
          cloudlog.info(".git directory changed, recreating overlay")
          overlay_initialized = False

      if not overlay_initialized:
        init_ovfs()
        overlay_initialized = True

      if params.get("IsOffroad") == b"1":
        update_available = attempt_update(wait_helper) or update_available
        update_failed_count = 0
        if not update_available and os.path.isdir(NEOSUPDATE_DIR):
          shutil.rmtree(NEOSUPDATE_DIR)
      else:
        cloudlog.info("not running updater, openpilot running")

    except subprocess.CalledProcessError as e:
      cloudlog.event(
        "update process failed",
        cmd=e.cmd,
        output=e.output,
        returncode=e.returncode
      )
      exception = e
      overlay_initialized = False
    except Exception:
      cloudlog.exception("uncaught updated exception, shouldn't happen")

    params.put("UpdateFailedCount", str(update_failed_count))
    if exception is None:
      params.delete("LastUpdateException")
    else:
      params.put("LastUpdateException", f"command failed: {exception.cmd}\n{exception.output}")

    # Wait 10 minutes between update attempts
    wait_helper.sleep(60*10)

  dismount_ovfs()

if __name__ == "__main__":
  main()
