#!/usr/bin/env python3
import os
import subprocess
from typing import List, Optional
from functools import lru_cache

from openpilot.common.basedir import BASEDIR
from openpilot.system.swaglog import cloudlog

RELEASE_BRANCHES = ['release3-staging', 'dashcam3-staging', 'release3', 'dashcam3', 'nightly']
TESTED_BRANCHES = RELEASE_BRANCHES + ['devel', 'devel-staging']

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
def get_normalized_origin(default: Optional[str] = None) -> Optional[str]:
  origin: Optional[str] = get_origin()

  if origin is None:
    return default

  return origin.replace("git@", "", 1) \
               .replace(".git", "", 1) \
               .replace("https://", "", 1) \
               .replace(":", "/", 1)


@cache
def get_version() -> str:
  with open(os.path.join(BASEDIR, "common", "version.h")) as _versionf:
    version = _versionf.read().split('"')[1]
  return version

@cache
def get_short_version() -> str:
  return get_version().split('-')[0]  # type: ignore

@cache
def is_prebuilt() -> bool:
  return os.path.exists(os.path.join(BASEDIR, 'prebuilt'))


@cache
def is_comma_remote() -> bool:
  # note to fork maintainers, this is used for release metrics. please do not
  # touch this to get rid of the orange startup alert. there's better ways to do that
  origin: Optional[str] = get_origin()
  if origin is None:
    return False

  return origin.startswith(('git@github.com:commaai', 'https://github.com/commaai'))


@cache
def is_tested_branch() -> bool:
  return get_short_branch() in TESTED_BRANCHES

@cache
def is_release_branch() -> bool:
  return get_short_branch() in RELEASE_BRANCHES

@cache
def is_dirty() -> bool:
  origin = get_origin()
  branch = get_branch()
  if (origin is None) or (branch is None):
    return True

  dirty = False
  try:
    # Actually check dirty files
    if not is_prebuilt():
      # This is needed otherwise touched files might show up as modified
      try:
        subprocess.check_call(["git", "update-index", "--refresh"])
      except subprocess.CalledProcessError:
        pass

      dirty = (subprocess.call(["git", "diff-index", "--quiet", branch, "--"]) != 0)
  except subprocess.CalledProcessError:
    cloudlog.exception("git subprocess failed while checking dirty")
    dirty = True

  return dirty


if __name__ == "__main__":
  from openpilot.common.params import Params

  params = Params()
  params.put("TermsVersion", terms_version)
  params.put("TrainingVersion", training_version)

  print(f"Dirty: {is_dirty()}")
  print(f"Version: {get_version()}")
  print(f"Short version: {get_short_version()}")
  print(f"Origin: {get_origin()}")
  print(f"Normalized origin: {get_normalized_origin()}")
  print(f"Branch: {get_branch()}")
  print(f"Short branch: {get_short_branch()}")
  print(f"Prebuilt: {is_prebuilt()}")
