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
# The swap on boot is triggered by /data/data/com.termux/files/continue.sh,
# gated on the existence of $FINALIZED/.overlay_consistent and also the
# existence and mtime of $BASEDIR/.overlay_canary.


import os
import datetime
import subprocess
from stat import S_ISREG, S_ISDIR, S_ISLNK, S_IMODE, ST_MODE, ST_INO, ST_UID, ST_GID, ST_ATIME, ST_MTIME
import shutil
import time
import signal
from pathlib import Path
import fcntl
from cffi import FFI

from common.basedir import BASEDIR
from common.params import Params
from selfdrive.swaglog import cloudlog

STAGING_ROOT = "/data/safe_staging"

OVERLAY_UPPER = os.path.join(STAGING_ROOT, "upper")
OVERLAY_METADATA = os.path.join(STAGING_ROOT, "metadata")
OVERLAY_MERGED = os.path.join(STAGING_ROOT, "merged")
FINALIZED = os.path.join(STAGING_ROOT, "finalized")

NICE_LOW_PRIORITY = ["nice", "-n", "19"]
SHORT = os.getenv("SHORT") is not None

class SignalHandler:
  continue_running = True  # Reset to False on receipt of SIGINT or SIGTERM

  def __init__(self):
    signal.signal(signal.SIGTERM, self.graceful_shutdown)
    signal.signal(signal.SIGINT, self.graceful_shutdown)
    signal.signal(signal.SIGHUP, self.update_now)

  def graceful_shutdown(self, signum, frame):
    cloudlog.info(f"caught SIGINT/SIGTERM, graceful shutdown in progress")
    self.continue_running = False

  def update_now(self, signum, frame):
    # Just returns, having interrupted the sleep() in wait_between_updates()
    cloudlog.info(f"caught SIGHUP, running update check immediately")

# Workaround for the EON/termux build of Python having os.link removed.
ffi = FFI()
ffi.cdef("int link(const char *oldpath, const char *newpath);")
libc = ffi.dlopen(None)

# **** helper functions ****

def wait_between_updates():
  if SHORT:
    time.sleep(10)
  else:
    time.sleep(60 * 10)

def link(src, dest):
  # Workaround for the EON/termux build of Python having os.link removed.
  return libc.link(src.encode(), dest.encode())

def run(cmd, cwd=None):
  return subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.STDOUT, encoding='utf8')

# **** meaty functions ****

def init_ovfs():
  if os.path.ismount(OVERLAY_MERGED):
    cloudlog.error("cleaning up stale overlay mount")
    run(["umount", OVERLAY_MERGED])

  cloudlog.info("preparing new safe staging area")
  if os.path.isdir(STAGING_ROOT):
    shutil.rmtree(STAGING_ROOT)

  for dirname in [STAGING_ROOT, OVERLAY_UPPER, OVERLAY_METADATA, OVERLAY_MERGED, FINALIZED]:
    os.mkdir(dirname, 0o755)
  if not os.lstat(BASEDIR).st_dev == os.lstat(OVERLAY_MERGED).st_dev:
    raise RuntimeError("base and overlay merge directories are on different filesystems; not valid for overlay FS!")

  if os.path.isfile(os.path.join(BASEDIR, ".overlay_consistent")):
    os.remove(os.path.join(BASEDIR, ".overlay_consistent"))

  # Leave a timestamped canary in BASEDIR to check at startup. If the canary
  # disappears, or critical mtimes in BASEDIR are newer than the canary,
  # continue.sh assumes that BASEDIR is being modified or used for local
  # development, and discards the overlay.
  Path(os.path.join(BASEDIR, ".overlay_canary")).touch()

  overlay_opts = f"lowerdir={BASEDIR},upperdir={OVERLAY_UPPER},workdir={OVERLAY_METADATA}"
  run(["mount", "-t", "overlay", "-o", overlay_opts, "none", OVERLAY_MERGED])

def inodes_in_tree(search_dir):
  # Given a search root, produce a dictionary mapping of inodes to relative
  # pathnames of regular files (no directories, symlinks, or special files).
  inode_map = {}
  for root, dirs, files in os.walk(search_dir, topdown=True):
    for file_name in files:
      full_path_name = os.path.join(root, file_name)
      st = os.lstat(full_path_name)
      if S_ISREG(st[ST_MODE]):
        # Debugging
        print(f"adding to inode tree: {st[ST_INO]} - {os.path.relpath(full_path_name, search_dir)}")
        inode_map[st[ST_INO]] = os.path.relpath(full_path_name, search_dir)
  return inode_map

def dup_ovfs_object(inode_map, source_obj, target_dir):
  # Given a relative pathname to copy, and a new target root, duplicate the
  # source object in the target root, using hardlinks for regular files.

  source_full_path = os.path.join(OVERLAY_MERGED, source_obj)
  st = os.lstat(source_full_path)
  target_obj = os.path.join(target_dir, source_obj)

  # Debugging
  print(f"dup_ovfs_object: source_obj {source_obj} target_dir {target_dir} target_obj {target_obj}")

  if S_ISREG(st[ST_MODE]):
    # Hardlink all regular files; ownership and permissions are shared.
    link(inode_map[st[ST_INO]], target_obj)
  else:
    # Recreate all directories and symlinks; copy ownership and permissions.
    if S_ISDIR(st[ST_MODE]):
      os.mkdir(os.path.join(FINALIZED, source_obj), S_IMODE(st[ST_MODE]))
    elif S_ISLNK(st[ST_MODE]):
      os.symlink(os.readlink(source_obj), target_obj)
      os.chmod(target_obj, S_IMODE(st[ST_MODE]), follow_symlinks=False)
    else:
      # Ran into a FIFO, socket, etc. Should not happen in OP install dir.
      # Ignore without copying for the time being; revisit later if needed.
      pass
    os.chown(target_obj, st[ST_UID], st[ST_GID], follow_symlinks=False)

  # Sync target mtimes to the cached lstat() value from each source object.
  # Restores shared inode mtimes after linking, fixes symlinks and dirs.
  os.utime(target_obj, (st[ST_ATIME], st[ST_MTIME]), follow_symlinks=False)

def finalize_from_ovfs():
  # Take the current OverlayFS merged view and finalize a copy outside of
  # OverlayFS, ready to be swapped-in at BASEDIR.
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

def attempt_update():
  cloudlog.info("attempting git update inside staging overlay")

  # Un-set the validity flag to prevent the finalized tree from being
  # activated later if the update attempt is interrupted.
  if os.path.isfile(os.path.join(FINALIZED, ".overlay_consistent")):
    os.remove(os.path.join(FINALIZED, ".overlay_consistent"))
  os.system("sync")

  r = run(NICE_LOW_PRIORITY + ["git", "fetch"], OVERLAY_MERGED)
  cloudlog.info("git fetch success: %s", r)

  cur_hash = run(["git", "rev-parse", "HEAD"], OVERLAY_MERGED).rstrip()
  upstream_hash = run(["git", "rev-parse", "@{u}"], OVERLAY_MERGED).rstrip()

  cloudlog.info("comparing %s to %s" % (cur_hash, upstream_hash))
  if cur_hash != upstream_hash:
    cloudlog.info("git reset in progress")
    r = [run(NICE_LOW_PRIORITY + ["git", "reset", "--hard", "@{u}"], OVERLAY_MERGED),
         run(NICE_LOW_PRIORITY + ["git", "clean", "-xdf"], OVERLAY_MERGED),
         run(NICE_LOW_PRIORITY + ["git", "submodule", "init"], OVERLAY_MERGED),
         run(NICE_LOW_PRIORITY + ["git", "submodule", "update"], OVERLAY_MERGED)]
    cloudlog.info("git reset success: %s", '\n'.join(r))
    # TODO: scons prebuild in background (offroad only, preferably interruptible if ignition comes on)
    # TODO: NEOS download in background (can and probably should do this outside the overlay)
    finalize_from_ovfs()
    update_params(with_date=True, new_version=True)
    cloudlog.info("update successful!")
  else:
    update_params(with_date=True, new_version=False)
    cloudlog.info("nothing new from git at this time")

  # Make sure the validity flag lands on disk LAST, only when the local git
  # repo and OP install are in a consistent state.
  os.system("sync")
  Path(os.path.join(FINALIZED, ".overlay_consistent")).touch()
  os.system("sync")

def update_params(with_date=False, new_version=False):
  params = Params()
  if new_version:
    try:
      with open(os.path.join(FINALIZED, "RELEASES.md"), "rb") as f:
        r = f.read()
      r = r[:r.find(b'\n\n')]  # Slice latest release notes
      params.put("ReleaseNotes", r + b"\n")
    except Exception:
      params.put("ReleaseNotes", "")
    params.put("UpdateAvailable", "1")
  else:
    params.put("UpdateAvailable", "0")

  if with_date:
    t = datetime.datetime.now().isoformat()
    params.put("LastUpdateTime", t.encode('utf8'))

# **** main loop ****

def main(gctx=None):
  overlay_init_done = False

  if not os.geteuid() == 0:
    raise RuntimeError("updated must be launched as root!")

  ov_lock_fd = open('/tmp/safe_staging_overlay.lock', 'w')
  try:
    fcntl.flock(ov_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
  except IOError:
    raise RuntimeError("couldn't get overlay lock; is another updated running?")

  while SignalHandler.continue_running:
    time_wrong = datetime.datetime.now().year < 2019
    ping_failed = subprocess.call(["ping", "-W", "4", "-c", "1", "8.8.8.8"])
    if ping_failed or time_wrong:
      wait_between_updates()
      continue

    try:
      # Wait until we have a valid datetime to initialize the overlay
      if not overlay_init_done:
        init_ovfs()

      attempt_update()

    except subprocess.CalledProcessError as e:
      cloudlog.event("update process failed",
        cmd=e.cmd,
        output=e.output,
        returncode=e.returncode)
      return False
    except Exception:
      cloudlog.exception("uncaught updated exception, shouldn't happen")

    wait_between_updates()

  # SignalHandler caught SIGINT or SIGTERM
  run(["umount", OVERLAY_MERGED])
  cloudlog.info("graceful shutdown complete!")


if __name__ == "__main__":
  # Commit noise to test updates 9283234534242 asdfasdfa sdfasd
  main()
