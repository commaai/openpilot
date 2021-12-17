#!/usr/bin/env python3
import os
import subprocess
from typing import List, Optional
from functools import lru_cache

from common.basedir import BASEDIR
from selfdrive.swaglog import cloudlog


TESTED_BRANCHES = ['devel', 'release2-staging', 'release3-staging', 'dashcam-staging', 'release2', 'release3', 'dashcam']
FORK_BRANCHES = ['stock_additions', 'SA-master', 'SA-master-acados']  # tested SA branches
FORK_BRANCHES += [f'{prefix}_{brnch}' for brnch in FORK_BRANCHES for prefix in ['shanesmiskol', 'sshane']]  # usernames

training_version: bytes = b"0.2.0"
terms_version: bytes = b"2"


def cache(user_function, /):
  return lru_cache(maxsize=None)(user_function)


def run_cmd(cmd: List[str]) -> str:
  return subprocess.check_output(cmd, encoding='utf8').strip()


def run_cmd_default(cmd: List[str], default: Optional[str] = None) -> Optional[str]:
  try:
    return run_cmd(cmd)
  except subprocess.CalledProcessError:
    return default


@cache
def get_commit(branch: str = "HEAD", default: Optional[str] = None) -> Optional[str]:
  return run_cmd_default(["git", "rev-parse", branch], default=default)


@cache
def get_short_branch(default: Optional[str] = None) -> Optional[str]:
  return run_cmd_default(["git", "rev-parse", "--abbrev-ref", "HEAD"], default=default)


@cache
def get_branch(default: Optional[str] = None) -> Optional[str]:
  return run_cmd_default(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], default=default)


@cache
def get_origin(default: Optional[str] = None) -> Optional[str]:
  try:
    local_branch = run_cmd(["git", "name-rev", "--name-only", "HEAD"])
    tracking_remote = run_cmd(["git", "config", "branch." + local_branch + ".remote"])
    return run_cmd(["git", "config", "remote." + tracking_remote + ".url"])
  except subprocess.CalledProcessError:  # Not on a branch, fallback
    return run_cmd_default(["git", "config", "--get", "remote.origin.url"], default=default)


@cache
def get_version() -> str:
  with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "common", "version.h")) as _versionf:
    version = _versionf.read().split('"')[1]
  return version


@cache
def get_prebuilt() -> bool:
  return os.path.exists(os.path.join(BASEDIR, 'prebuilt'))


@cache
def get_comma_remote() -> bool:
  origin = get_origin()
  if origin is None:
    return False

  return origin.startswith('git@github.com:commaai') or origin.startswith('https://github.com/commaai')

@cache
def get_fork_remote() -> bool:
  origin = get_origin()
  if origin is None:
    return False

  return origin.startswith('git@github.com:sshane') or origin.startswith('https://github.com/sshane')


@cache
def get_tested_branch() -> bool:
  return get_short_branch() in TESTED_BRANCHES

@cache
def get_fork_tested_branch() -> bool:
  return get_short_branch() in FORK_BRANCHES


# By default, this behaves as comma intends
# If passed True however, it returns if the user is on a dirty fork install
@cache
def get_dirty(fork=False) -> bool:
  origin = get_origin()
  branch = get_branch()
  if (origin is None) or (branch is None):
    return True

  dirty = False
  try:
    # Actually check dirty files
    if not get_prebuilt():
      # This is needed otherwise touched files might show up as modified
      try:
        subprocess.check_call(["git", "update-index", "--refresh"])
      except subprocess.CalledProcessError:
        pass

      dirty = (subprocess.call(["git", "diff-index", "--quiet", branch, "--"]) != 0)

      # Log dirty files
      if dirty and get_comma_remote():
        try:
          dirty_files = run_cmd(["git", "diff-index", branch, "--"])
          cloudlog.event("dirty comma branch", version=get_version(), dirty=dirty, origin=origin, branch=branch,
                          dirty_files=dirty_files, commit=get_commit(), origin_commit=get_commit(branch))
        except subprocess.CalledProcessError:
          pass

    correct_remote = get_comma_remote() if not fork else get_fork_remote()
    dirty = dirty or (not correct_remote)

  except subprocess.CalledProcessError:
    cloudlog.exception("git subprocess failed while checking dirty")
    dirty = True

  return dirty


if __name__ == "__main__":
  from common.params import Params

  params = Params()
  params.put("TermsVersion", terms_version)
  params.put("TrainingVersion", training_version)

  print("Dirty: %s" % get_dirty())
  print("Version: %s" % get_version())
  print("Origin: %s" % get_origin())
  print("Branch: %s" % get_branch())
  print("Short branch: %s" % get_short_branch())
  print("Prebuilt: %s" % get_prebuilt())
