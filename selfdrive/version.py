#!/usr/bin/env python3
import os
import subprocess
from selfdrive.swaglog import cloudlog


def get_git_commit():
  try:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], encoding='utf8').strip()
  except subprocess.CalledProcessError:
    return None


def get_git_branch():
  try:
    return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], encoding='utf8').strip()
  except subprocess.CalledProcessError:
    return None


def get_git_full_branchname():
  try:
    return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], encoding='utf8').strip()
  except subprocess.CalledProcessError:
    return None


def get_git_remote():
  try:
    local_branch = subprocess.check_output(["git", "name-rev", "--name-only", "HEAD"], encoding='utf8').strip()
    tracking_remote = subprocess.check_output(["git", "config", "branch." + local_branch + ".remote"], encoding='utf8').strip()
    return subprocess.check_output(["git", "config", "remote." + tracking_remote + ".url"], encoding='utf8').strip()

  except subprocess.CalledProcessError:
    try:
      # Not on a branch, fallback
      return subprocess.check_output(["git", "config", "--get", "remote.origin.url"], encoding='utf8').strip()
    except subprocess.CalledProcessError:
      return None


with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "common", "version.h")) as _versionf:
  version = _versionf.read().split('"')[1]

training_version = b"0.2.0"
terms_version = b"2"

dirty = True
origin = get_git_remote()
branch = get_git_full_branchname()

try:
  # This is needed otherwise touched files might show up as modified
  try:
    subprocess.check_call(["git", "update-index", "--refresh"])
  except subprocess.CalledProcessError:
    pass

  if (origin is not None) and (branch is not None):
    comma_remote = origin.startswith('git@github.com:commaai') or origin.startswith('https://github.com/commaai')

    dirty = not comma_remote
    dirty = dirty or ('master' in branch)
    dirty = dirty or (subprocess.call(["git", "diff-index", "--quiet", branch, "--"]) != 0)

    if dirty:
      dirty_files = subprocess.check_output(["git", "diff-index", branch, "--"], encoding='utf8')
      commit = subprocess.check_output(["git", "rev-parse", "--verify", "HEAD"], encoding='utf8').rstrip()
      origin_commit = subprocess.check_output(["git", "rev-parse", "--verify", branch], encoding='utf8').rstrip()
      cloudlog.event("dirty comma branch", version=version, dirty=dirty, origin=origin, branch=branch, dirty_files=dirty_files, commit=commit, origin_commit=origin_commit)

except subprocess.CalledProcessError:
  dirty = True
  cloudlog.exception("git subprocess failed while checking dirty")


if __name__ == "__main__":
  print("Dirty: %s" % dirty)
  print("Version: %s" % version)
  print("Remote: %s" % origin)
  print("Branch %s" % branch)
