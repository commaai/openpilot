#!/usr/bin/env python3

# Safe Update: A simple service that waits for network access and tries to
# update every 10 minutes. It's intended to make the OP update process more
# robust in the face of local Git repository or EON filesystem corruption.
#
# During normal operation, both onroad and offroad, the update process makes
# no changes to the BASEDIR install of OP. All update attempts are performed
# in a disposable staging area provided by OverlayFS.
#
# If an update succeeds, a flag is set, and the update is swapped in at the
# next reboot. If an update is interrupted or otherwise fails, the OverlayFS
# upper layer and metadata can be discarded before trying again.
#
# The swap on boot is triggered by /data/data/com.termux/files/continue.sh,
# gated on the existence of $FINALIZED/.update_succeeded.

# Short term roadmap:
# FIXME: Handle case of Git being corrupt before we even start (cloudlog git fsck and then re-clone?)
# (INPROG) FIXME: make sure updated is reentry safe (must be able to tolerate CLI invocation while running as daemon)
# TODO: design change: need to finalize after reboot as part of the switch process, will fix several issues below
# TODO: consider impact of SIGINT/SIGTERM and whether we should catch and dismount/finalize/etc?
# TODO: explore doing git gc, and/or limited-depth clones

# Long term roadmap:
# TODO: test suite to compare merged-to-finalized, even though manual compare looks good now
# TODO: scons prebuild the update, maybe from manager so it can run offroad-only and be interrupted at onroad
# TODO: download any required NEOS update in the background


import os
import datetime
import subprocess
from stat import *
import shutil
import time
from pathlib import Path
import fcntl
from cffi import FFI

from common.basedir import BASEDIR
from common.params import Params
from selfdrive.swaglog import cloudlog

STAGING_ROOT = "/data/safe_staging"
OVERLAY_LOCK = "/tmp/safe_staging_overlay.lock"
UPDATE_LOCK = "/tmp/safe_staging_update.lock"

OVERLAY_UPPER = os.path.join(STAGING_ROOT, "upper")
OVERLAY_METADATA = os.path.join(STAGING_ROOT, "metadata")
OVERLAY_MERGED = os.path.join(STAGING_ROOT, "merged")
FINALIZED = os.path.join(STAGING_ROOT, "finalized")

NICE_LOW_PRIORITY = ["nice", "-n", "19"]
SHORT = os.getenv("SHORT") is not None

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
    # We had to have the overlay lock to get this far, so this must be a stale
    # mount from a terminated updated process.
    run(["umount", OVERLAY_MERGED])

  cloudlog.info("preparing new staging area")
  if os.path.isfile(os.path.join(BASEDIR, ".update_succeeded")):
    os.remove(os.path.join(BASEDIR, ".update_succeeded"))
  if os.path.isdir(STAGING_ROOT):
    shutil.rmtree(STAGING_ROOT)

  for dirname in [STAGING_ROOT, OVERLAY_UPPER, OVERLAY_METADATA, OVERLAY_MERGED, FINALIZED]:
    os.mkdir(dirname, 0o755)
  if not os.lstat(BASEDIR).st_dev == os.lstat(STAGING_ROOT).st_dev:
    cloudlog.error("base and staging directories are on different filesystems, not valid for overlay FS!")
    exit(1)

  overlay_opts = f"lowerdir={BASEDIR},upperdir={OVERLAY_UPPER},workdir={OVERLAY_METADATA}"
  run(["mount", "-t", "overlay", "-o", overlay_opts, "none", OVERLAY_MERGED])

def inodes_in_tree(search_dir):
  # Given a search root, produce a dictionary mapping of inodes to relative
  # pathnames of regular files (no directories, symlinks, or special files).
  inode_map = {}
  os.chdir(search_dir)
  for root, dirs, files in os.walk('.', topdown=True):
    for file_name in files:
      full_name = os.path.join(search_dir, root, file_name)
      st = os.stat(full_name)
      if S_ISREG(st[ST_MODE]):
        inode_map.update({st[ST_INO]: full_name})
  return inode_map

def dup_ovfs_object(inode_map, source_obj, target_dir):
  # Given a relative pathname to copy, and a new target root, duplicate the
  # source object in the target root, using hardlinks for regular files.

  st = os.lstat(source_obj)
  target_obj = os.path.join(target_dir, source_obj)

  if S_ISREG(st[ST_MODE]):
    # Hardlink all regular files; ownership and permissions are inherited.
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
      # Ignore without copying for the time being, revisit if necessary.
      pass
    os.chown(target_obj, st[ST_UID], st[ST_GID], follow_symlinks=False)
    os.utime(target_obj, (st[ST_ATIME], st[ST_MTIME]), follow_symlinks=False)

def finalize_from_ovfs():
  # Take the current OverlayFS merged view and finalize a copy outside of
  # OverlayFS, ready to be swapped-in at BASEDIR.
  #
  # The "copy" is done with hardlinks, but since the OverlayFS merge looks
  # like a different filesystem, and hardlinks can't cross filesystems, we
  # have to link from a pathname in the upper or lower layers.

  cloudlog.info("creating overlay inode index")
  inode_map = inodes_in_tree(BASEDIR)
  inode_map.update(inodes_in_tree(OVERLAY_UPPER))

  cloudlog.info("creating finalized version of the overlay")
  shutil.rmtree(FINALIZED)
  os.umask(0o077)
  os.mkdir(FINALIZED)
  os.chdir(OVERLAY_MERGED)
  for root, dirs, files in os.walk('.', topdown=True):
    for obj_name in dirs:
      dup_ovfs_object(inode_map, os.path.join(root, obj_name), FINALIZED)
    for obj_name in files:
      dup_ovfs_object(inode_map, os.path.join(root, obj_name), FINALIZED)
  os.chdir(BASEDIR)  # otherwise umount will fail later with "target is busy"

def attempt_update():
  upd_lock_fd = open(UPDATE_LOCK, 'r')
  try:
    fcntl.flock(upd_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
  except OSError:
    print("couldn't get update lock; is another updated running? going back to sleep...")
    upd_lock_fd.close()
    return

  cloudlog.info("attempting git update inside staging overlay")
  os.chdir(OVERLAY_MERGED)

  # Un-set the successful flag to prevent the finalized tree from being
  # activated later if the update fails in some way.
  if os.path.isfile(os.path.join(FINALIZED, ".update_succeeded")):
    os.remove(os.path.join(FINALIZED, ".update_succeeded"))
  os.system("sync")

  r = run(NICE_LOW_PRIORITY + ["git", "fetch"], OVERLAY_MERGED)
  cloudlog.info("git fetch success: %s", r)

  cur_hash = run(["git", "rev-parse", "HEAD"], OVERLAY_MERGED).rstrip()
  upstream_hash = run(["git", "rev-parse", "@{u}"], OVERLAY_MERGED).rstrip()

  cloudlog.info("comparing %s to %s" % (cur_hash, upstream_hash))
  if cur_hash != upstream_hash:
    cloudlog.info("git reset in progress")
    r = [run(NICE_LOW_PRIORITY + ["git", "reset", "--hard", "@{u}"], OVERLAY_MERGED),
         # TODO: with scons, is clean still required?
         run(NICE_LOW_PRIORITY + ["git", "clean", "-xdf"], OVERLAY_MERGED),
         run(NICE_LOW_PRIORITY + ["git", "submodule", "init"], OVERLAY_MERGED),
         run(NICE_LOW_PRIORITY + ["git", "submodule", "update"], OVERLAY_MERGED)]
    cloudlog.info("git reset success: %s", '\n'.join(r))

    cloudlog.info("update successful!")
  else:
    cloudlog.info("nothing new from git at this time")

  # Make sure the update succeeded flag lands on disk LAST, only when the
  # local git repo and OP install state are in a consistent state.
  os.system("sync")
  Path(os.path.join(OVERLAY_MERGED, ".update_succeeded")).touch()
  os.system("sync")

  fcntl.flock(upd_lock_fd, fcntl.LOCK_UN)
  upd_lock_fd.close()

def update_params(with_date=False):
  params = Params()
  if os.path.isfile(os.path.join(FINALIZED, ".update_succeeded")):
    try:
      # TODO: This shouldn't be binary
      with open(os.path.join(FINALIZED, "RELEASES.md"), "rb") as f:
        r = f.read()
      r = r[:r.find(b'\n\n')] # Slice latest release notes
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
  if not os.geteuid() == 0:
    cloudlog.error("updated must be launched as root!")
    exit(1)

  ov_lock_fd = open(OVERLAY_LOCK, 'r')
  try:
    fcntl.flock(ov_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    init_ovfs()
  except IOError:
    print("couldn't get overlay lock; is another updated running? skipping init and continuing in update-only mode...")

  update_params()

  while True:
    time_wrong = datetime.datetime.now().year < 2019
    ping_failed = subprocess.call(["ping", "-W", "4", "-c", "1", "8.8.8.8"])
    if ping_failed or time_wrong:
      wait_between_updates()
      continue

    try:
      attempt_update()
      update_params(True)
    except subprocess.CalledProcessError as e:
      cloudlog.event("update process failed",
        cmd=e.cmd,
        output=e.output,
        returncode=e.returncode)
      return False
    except Exception:
      cloudlog.exception("uncaught updated exception, shouldn't happen")

    wait_between_updates()

if __name__ == "__main__":
  # Commit noise to test updates 9283597823atrsfdw324323e1y
  main()

