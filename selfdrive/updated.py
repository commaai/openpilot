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
import threading
from cffi import FFI
from pathlib import Path
from stat import S_ISREG, S_ISDIR, S_ISLNK, S_IMODE, \
                 ST_MODE, ST_INO, ST_UID, ST_GID, \
                 ST_ATIME, ST_MTIME

from common.basedir import BASEDIR
from common.params import Params
from selfdrive.swaglog import cloudlog

LOCK_FILE = "/tmp/safe_staging_overlay.lock"
STAGING_ROOT = "/data/safe_staging"
if os.getenv("UPDATER_TESTING") is not None:
  LOCK_FILE = os.getenv("UPDATER_LOCK_FILE", LOCK_FILE)
  STAGING_ROOT = os.getenv("UPDATER_STAGING_ROOT", STAGING_ROOT)

OVERLAY_UPPER = os.path.join(STAGING_ROOT, "upper")
OVERLAY_METADATA = os.path.join(STAGING_ROOT, "metadata")
OVERLAY_MERGED = os.path.join(STAGING_ROOT, "merged")
FINALIZED = os.path.join(STAGING_ROOT, "finalized")

NICE_LOW_PRIORITY = ["nice", "-n", "19"]

# Workaround for the EON/termux build of Python having os.link removed.
ffi = FFI()
ffi.cdef("int link(const char *oldpath, const char *newpath);")
libc = ffi.dlopen(None)


def link(src, dest):
  # Workaround for the EON/termux build of Python having os.link removed.
  return libc.link(src.encode(), dest.encode())


def run(cmd, cwd=None):
  return subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.STDOUT, encoding='utf8')


def remove_consistent_flag():
  os.system("sync")
  consistent_file = Path(os.path.join(FINALIZED, ".overlay_consistent"))
  if os.path.isfile(consistent_file):
    consistent_file.unlink()
  os.system("sync")


def set_consistent_flag():
  print("setting consistent flag")
  consistent_file = Path(os.path.join(FINALIZED, ".overlay_consistent"))
  os.system("sync")
  consistent_file.touch()
  os.system("sync")
  print("done setting consistent flag")


def set_update_available_params(new_version=False):
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
  # We sync FS object atimes (which EON doesn't use) and mtimes, but ctimes
  # are outside user control. Make sure Git is set up to ignore system ctimes,
  # because they change when we make hard links during finalize. Otherwise,
  # there is a lot of unnecessary churn. This appears to be a common need on
  # OSX as well: https://www.git-tower.com/blog/make-git-rebase-safe-on-osx/
  try:
    trustctime = run(["git", "config", "--get", "core.trustctime"], cwd)
    trustctime_set = (trustctime.strip() == "false")
  except subprocess.CalledProcessError:
    trustctime_set = False

  if not trustctime_set:
    cloudlog.info("Setting core.trustctime false")
    run(["git", "config", "core.trustctime", "false"], cwd)

  # We are temporarily using copytree to copy the directory, which also changes
  # inode numbers. Ignore those changes too.
  try:
    checkstat = run(["git", "config", "--get", "core.checkStat"], cwd)
    checkstat_set = (checkstat.strip() == "minimal")
  except subprocess.CalledProcessError:
    checkstat_set = False

  if not checkstat_set:
    cloudlog.info("Setting core.checkState minimal")
    run(["git", "config", "core.checkStat", "minimal"], cwd)


def init_ovfs():
  cloudlog.info("preparing new safe staging area")
  Params().put("UpdateAvailable", "0")

  remove_consistent_flag()

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

  # Leave a timestamped canary in BASEDIR to check at startup. The EON clock
  # should be correct by the time we get here. If the init file disappears, or
  # critical mtimes in BASEDIR are newer than .overlay_init, continue.sh can
  # assume that BASEDIR has used for local development or otherwise modified,
  # and skips the update activation attempt.
  Path(os.path.join(BASEDIR, ".overlay_init")).touch()

  overlay_opts = f"lowerdir={BASEDIR},upperdir={OVERLAY_UPPER},workdir={OVERLAY_METADATA}"
  run(["mount", "-t", "overlay", "-o", overlay_opts, "none", OVERLAY_MERGED])


def inodes_in_tree(search_dir):
  """Given a search root, produce a dictionary mapping of inodes to relative
  pathnames of regular files (no directories, symlinks, or special files)."""
  inode_map = {}
  for root, _, files in os.walk(search_dir, topdown=True):
    for file_name in files:
      full_path_name = os.path.join(root, file_name)
      st = os.lstat(full_path_name)
      if S_ISREG(st[ST_MODE]):
        inode_map[st[ST_INO]] = full_path_name
  return inode_map


def dup_ovfs_object(inode_map, source_obj, target_dir):
  """Given a relative pathname to copy, and a new target root, duplicate the
  source object in the target root, using hardlinks for regular files."""

  source_full_path = os.path.join(OVERLAY_MERGED, source_obj)
  st = os.lstat(source_full_path)
  target_full_path = os.path.join(target_dir, source_obj)

  if S_ISREG(st[ST_MODE]):
    # Hardlink all regular files; ownership and permissions are shared.
    link(inode_map[st[ST_INO]], target_full_path)
  else:
    # Recreate all directories and symlinks; copy ownership and permissions.
    if S_ISDIR(st[ST_MODE]):
      os.mkdir(os.path.join(FINALIZED, source_obj), S_IMODE(st[ST_MODE]))
    elif S_ISLNK(st[ST_MODE]):
      os.symlink(os.readlink(source_full_path), target_full_path)
      os.chmod(target_full_path, S_IMODE(st[ST_MODE]), follow_symlinks=False)
    else:
      # Ran into a FIFO, socket, etc. Should not happen in OP install dir.
      # Ignore without copying for the time being; revisit later if needed.
      cloudlog.error("can't copy this file type: %s" % source_full_path)
    os.chown(target_full_path, st[ST_UID], st[ST_GID], follow_symlinks=False)

  # Sync target mtimes to the cached lstat() value from each source object.
  # Restores shared inode mtimes after linking, fixes symlinks and dirs.
  os.utime(target_full_path, (st[ST_ATIME], st[ST_MTIME]), follow_symlinks=False)


def finalize_from_ovfs_hardlink():
  """Take the current OverlayFS merged view and finalize a copy outside of
  OverlayFS, ready to be swapped-in at BASEDIR. Copy using hardlinks"""

  cloudlog.info("creating finalized version of the overlay")

  # The "copy" is done with hardlinks, but since the OverlayFS merge looks
  # like a different filesystem, and hardlinks can't cross filesystems, we
  # have to borrow a source pathname from the upper or lower layer.
  inode_map = inodes_in_tree(BASEDIR)
  inode_map.update(inodes_in_tree(OVERLAY_UPPER))

  shutil.rmtree(FINALIZED)
  os.umask(0o077)
  os.mkdir(FINALIZED)
  for root, dirs, files in os.walk(OVERLAY_MERGED, topdown=True):
    for obj_name in dirs:
      relative_path_name = os.path.relpath(os.path.join(root, obj_name), OVERLAY_MERGED)
      dup_ovfs_object(inode_map, relative_path_name, FINALIZED)
    for obj_name in files:
      relative_path_name = os.path.relpath(os.path.join(root, obj_name), OVERLAY_MERGED)
      dup_ovfs_object(inode_map, relative_path_name, FINALIZED)
  cloudlog.info("done finalizing overlay")


def finalize_from_ovfs_copy():
  """Take the current OverlayFS merged view and finalize a copy outside of
  OverlayFS, ready to be swapped-in at BASEDIR. Copy using shutil.copytree"""

  cloudlog.info("creating finalized version of the overlay")
  shutil.rmtree(FINALIZED)
  shutil.copytree(OVERLAY_MERGED, FINALIZED, symlinks=True)
  cloudlog.info("done finalizing overlay")


def attempt_update():
  cloudlog.info("attempting git update inside staging overlay")

  setup_git_options(OVERLAY_MERGED)

  git_fetch_output = run(NICE_LOW_PRIORITY + ["git", "fetch"], OVERLAY_MERGED)
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
        run(NICE_LOW_PRIORITY + ["git", "reset", "--hard", "@{u}"], OVERLAY_MERGED),
        run(NICE_LOW_PRIORITY + ["git", "clean", "-xdf"], OVERLAY_MERGED),
        run(NICE_LOW_PRIORITY + ["git", "submodule", "init"], OVERLAY_MERGED),
        run(NICE_LOW_PRIORITY + ["git", "submodule", "update"], OVERLAY_MERGED),
      ]
      cloudlog.info("git reset success: %s", '\n'.join(r))

    # Un-set the validity flag to prevent the finalized tree from being
    # activated later if the finalize step is interrupted
    remove_consistent_flag()

    finalize_from_ovfs_copy()

    # Make sure the validity flag lands on disk LAST, only when the local git
    # repo and OP install are in a consistent state.
    set_consistent_flag()

    cloudlog.info("update successful!")
  else:
    cloudlog.info("nothing new from git at this time")

  set_update_available_params(new_version=new_version)


def main():
  params = Params()

  if params.get("DisableUpdates") == b"1":
    raise RuntimeError("updates are disabled by param")

  if os.geteuid() != 0:
    raise RuntimeError("updated must be launched as root!")

  # Set low io priority
  p = psutil.Process()
  if psutil.LINUX:
    p.ionice(psutil.IOPRIO_CLASS_BE, value=7)

  ov_lock_fd = open(LOCK_FILE, 'w')
  try:
    fcntl.flock(ov_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
  except IOError:
    raise RuntimeError("couldn't get overlay lock; is another updated running?")

  # Setup a signal handler to immediately trigger an update
  ready_event = threading.Event()
  def set_ready(signum, frame):
    if not ready_event.set():
      cloudlog.info("caught SIGHUP, running update check immediately")
    ready_event.set()
  signal.signal(signal.SIGHUP, set_ready)

  # Setup the signal handlers to cleanly exit
  exit_event = threading.Event()
  def set_do_exit(signum, frame):
    # umount -f doesn't appear effective in avoiding "device busy" on NEOS,
    # so don't actually exit until the next convenient opportunity in main().
    cloudlog.info("caught SIGINT/SIGTERM, dismounting overlay at next opportunity")
    exit_event.set()
    ready_event.set()
  signal.signal(signal.SIGTERM, set_do_exit)
  signal.signal(signal.SIGINT, set_do_exit)

  # Wait for IsOffroad to be set
  ready_event.wait(timeout=30)

  update_failed_count = 0
  overlay_init_done = False
  while not exit_event.is_set():
    update_failed_count += 1
    ready_event.clear()

    # Check for internet every 30s
    time_wrong = datetime.datetime.utcnow().year < 2019
    ping_failed = os.system("ping -W 4 -c 1 8.8.8.8") != 0
    if ping_failed or time_wrong:
      ready_event.wait(30)
      continue

    # Attempt an update
    try:
      # Re-create the overlay if BASEDIR/.git has changed since we created the overlay
      if overlay_init_done:
        overlay_init_fn = os.path.join(BASEDIR, ".overlay_init")
        git_dir_path = os.path.join(BASEDIR, ".git")
        new_files = run(["find", git_dir_path, "-newer", overlay_init_fn])

        if len(new_files.splitlines()):
          cloudlog.info(".git directory changed, recreating overlay")
          overlay_init_done = False

      if not overlay_init_done:
        init_ovfs()
        overlay_init_done = True

      if params.get("IsOffroad") == b"1":
        attempt_update()
        update_failed_count = 0
      else:
        cloudlog.info("not running updater, openpilot running")

    except subprocess.CalledProcessError as e:
      cloudlog.event(
        "update process failed",
        cmd=e.cmd,
        output=e.output,
        returncode=e.returncode
      )
      overlay_init_done = False
    except Exception:
      cloudlog.exception("uncaught updated exception, shouldn't happen")

    params.put("UpdateFailedCount", str(update_failed_count))

    # Wait 10 minutes between update attempts
    ready_event.wait(60*10)

  # We've been signaled to shut down
  dismount_ovfs()

if __name__ == "__main__":
  main()
