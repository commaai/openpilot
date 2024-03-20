#!/usr/bin/env python3
import os
import subprocess


from openpilot.common.basedir import BASEDIR
from openpilot.common.swaglog import cloudlog
from openpilot.common.utils import cache
from openpilot.common.git import get_origin, get_branch, get_short_branch, get_normalized_origin, get_commit_date


RELEASE_BRANCHES = ['release3-staging', 'release3', 'nightly']
TESTED_BRANCHES = RELEASE_BRANCHES + ['devel', 'devel-staging']

training_version: bytes = b"0.2.0"
terms_version: bytes = b"2"


def get_version(path: str = BASEDIR) -> str:
  with open(os.path.join(path, "common", "version.h")) as _versionf:
    version = _versionf.read().split('"')[1]
  return version


def get_release_notes(path: str = BASEDIR) -> str:
  with open(os.path.join(path, "RELEASES.md"), "r") as f:
    return f.read().split('\n\n', 1)[0]


@cache
def get_short_version() -> str:
  return get_version().split('-')[0]

@cache
def is_prebuilt() -> bool:
  return os.path.exists(os.path.join(BASEDIR, 'prebuilt'))


@cache
def is_comma_remote() -> bool:
  # note to fork maintainers, this is used for release metrics. please do not
  # touch this to get rid of the orange startup alert. there's better ways to do that
  return get_normalized_origin() == "github.com/commaai/openpilot"

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
  if not origin or not branch:
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
  print(f"Commit date: {get_commit_date()}")
