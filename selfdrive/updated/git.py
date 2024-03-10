import datetime
import os
import re
import shutil
import subprocess
import time

from collections import defaultdict
from pathlib import Path
from typing import List

from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.updated.common import FINALIZED, STAGING_ROOT, UpdateStrategy, \
                                               get_consistent_flag, get_release_notes, get_version, set_consistent_flag, run


OVERLAY_UPPER = os.path.join(STAGING_ROOT, "upper")
OVERLAY_METADATA = os.path.join(STAGING_ROOT, "metadata")
OVERLAY_MERGED = os.path.join(STAGING_ROOT, "merged")
OVERLAY_INIT = Path(os.path.join(BASEDIR, ".overlay_init"))


def setup_git_options(cwd: str) -> None:
  # We sync FS object atimes (which NEOS doesn't use) and mtimes, but ctimes
  # are outside user control. Make sure Git is set up to ignore system ctimes,
  # because they change when we make hard links during finalize. Otherwise,
  # there is a lot of unnecessary churn. This appears to be a common need on
  # OSX as well: https://www.git-tower.com/blog/make-git-rebase-safe-on-osx/

  # We are using copytree to copy the directory, which also changes
  # inode numbers. Ignore those changes too.

  # Set protocol to the new version (default after git 2.26) to reduce data
  # usage on git fetch --dry-run from about 400KB to 18KB.
  git_cfg = [
    ("core.trustctime", "false"),
    ("core.checkStat", "minimal"),
    ("protocol.version", "2"),
    ("gc.auto", "0"),
    ("gc.autoDetach", "false"),
  ]
  for option, value in git_cfg:
    run(["git", "config", option, value], cwd)


def dismount_overlay() -> None:
  if os.path.ismount(OVERLAY_MERGED):
    cloudlog.info("unmounting existing overlay")
    run(["sudo", "umount", "-l", OVERLAY_MERGED])


def init_overlay() -> None:

  # Re-create the overlay if BASEDIR/.git has changed since we created the overlay
  if OVERLAY_INIT.is_file() and os.path.ismount(OVERLAY_MERGED):
    git_dir_path = os.path.join(BASEDIR, ".git")
    new_files = run(["find", git_dir_path, "-newer", str(OVERLAY_INIT)])
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
  OVERLAY_INIT.touch()

  os.sync()
  overlay_opts = f"lowerdir={BASEDIR},upperdir={OVERLAY_UPPER},workdir={OVERLAY_METADATA}"

  mount_cmd = ["mount", "-t", "overlay", "-o", overlay_opts, "none", OVERLAY_MERGED]
  run(["sudo"] + mount_cmd)
  run(["sudo", "chmod", "755", os.path.join(OVERLAY_METADATA, "work")])

  git_diff = run(["git", "diff"], OVERLAY_MERGED)
  params.put("GitDiff", git_diff)
  cloudlog.info(f"git diff output:\n{git_diff}")


class GitUpdateStrategy(UpdateStrategy):

  def init(self) -> None:
    init_overlay()

  def cleanup(self) -> None:
    OVERLAY_INIT.unlink(missing_ok=True)

  def sync_branches(self):
    excluded_branches = ('release2', 'release2-staging')

    output = run(["git", "ls-remote", "--heads"], OVERLAY_MERGED)

    self.branches = defaultdict(lambda: None)
    for line in output.split('\n'):
      ls_remotes_re = r'(?P<commit_sha>\b[0-9a-f]{5,40}\b)(\s+)(refs\/heads\/)(?P<branch_name>.*$)'
      x = re.fullmatch(ls_remotes_re, line.strip())
      if x is not None and x.group('branch_name') not in excluded_branches:
        self.branches[x.group('branch_name')] = x.group('commit_sha')

    return self.branches

  def get_available_channels(self) -> List[str]:
    self.sync_branches()
    return list(self.branches.keys())

  def update_ready(self) -> bool:
    if get_consistent_flag():
      hash_mismatch = self.get_commit_hash(BASEDIR) != self.branches[self.target_channel]
      branch_mismatch = self.get_branch(BASEDIR) != self.target_channel
      on_target_channel = self.get_branch(FINALIZED) == self.target_channel
      return ((hash_mismatch or branch_mismatch) and on_target_channel)
    return False

  def update_available(self) -> bool:
    if os.path.isdir(OVERLAY_MERGED) and len(self.get_available_channels()) > 0:
      hash_mismatch = self.get_commit_hash(OVERLAY_MERGED) != self.branches[self.target_channel]
      branch_mismatch = self.get_branch(OVERLAY_MERGED) != self.target_channel
      return hash_mismatch or branch_mismatch
    return False

  def get_branch(self, path: str) -> str:
    return run(["git", "rev-parse", "--abbrev-ref", "HEAD"], path).rstrip()

  def get_commit_hash(self, path) -> str:
    return run(["git", "rev-parse", "HEAD"], path).rstrip()

  def get_current_channel(self) -> str:
    return self.get_branch(BASEDIR)

  def current_channel(self) -> str:
    return self.get_branch(BASEDIR)

  def describe_branch(self, basedir) -> str:
    if not os.path.exists(basedir):
      return ""

    version = ""
    branch = ""
    commit = ""
    commit_date = ""
    try:
      branch = self.get_branch(basedir)
      commit = self.get_commit_hash(basedir)[:7]
      version = get_version(basedir)

      commit_unix_ts = run(["git", "show", "-s", "--format=%ct", "HEAD"], basedir).rstrip()
      dt = datetime.datetime.fromtimestamp(int(commit_unix_ts))
      commit_date = dt.strftime("%b %d")
    except Exception:
      cloudlog.exception("updater.get_description")
    return f"{version} / {branch} / {commit} / {commit_date}"

  def describe_current_channel(self) -> tuple[str, str]:
    return self.describe_branch(BASEDIR), get_release_notes(BASEDIR)

  def describe_ready_channel(self) -> tuple[str, str]:
    if self.update_ready():
      return self.describe_branch(FINALIZED), get_release_notes(FINALIZED)

    return "", ""

  def fetch_update(self):
    cloudlog.info("attempting git fetch inside staging overlay")

    setup_git_options(OVERLAY_MERGED)

    branch = self.target_channel
    git_fetch_output = run(["git", "fetch", "origin", branch], OVERLAY_MERGED)
    cloudlog.info("git fetch success: %s", git_fetch_output)

    cloudlog.info("git reset in progress")
    cmds = [
      ["git", "checkout", "--force", "--no-recurse-submodules", "-B", branch, "FETCH_HEAD"],
      ["git", "reset", "--hard"],
      ["git", "clean", "-xdff"],
      ["git", "submodule", "sync"],
      ["git", "submodule", "update", "--init", "--recursive"],
      ["git", "submodule", "foreach", "--recursive", "git", "reset", "--hard"],
    ]
    r = [run(cmd, OVERLAY_MERGED) for cmd in cmds]
    cloudlog.info("git reset success: %s", '\n'.join(r))

  def fetched_path(self):
    return str(OVERLAY_MERGED)

  def finalize_update(self) -> None:
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
    run(["git", "submodule", "foreach", "--recursive", "git", "reset", "--hard"], FINALIZED)

    cloudlog.info("Starting git cleanup in finalized update")
    t = time.monotonic()
    try:
      run(["git", "gc"], FINALIZED)
      run(["git", "lfs", "prune"], FINALIZED)
      cloudlog.event("Done git cleanup", duration=time.monotonic() - t)
    except subprocess.CalledProcessError:
      cloudlog.exception(f"Failed git cleanup, took {time.monotonic() - t:.3f} s")

    set_consistent_flag(True)
    cloudlog.info("done finalizing overlay")
