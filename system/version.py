#!/usr/bin/env python3
from dataclasses import dataclass
import json
import os
import pathlib
import subprocess


from openpilot.common.basedir import BASEDIR
from openpilot.common.swaglog import cloudlog
from openpilot.common.utils import cache
from openpilot.common.git import get_commit, get_origin, get_branch, get_short_branch, get_commit_date


RELEASE_BRANCHES = ['release3-staging', 'release3', 'nightly']
TESTED_BRANCHES = RELEASE_BRANCHES + ['devel', 'devel-staging']

BUILD_METADATA_FILENAME = "build.json"

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
def is_prebuilt(path: str = BASEDIR) -> bool:
  return os.path.exists(os.path.join(path, 'prebuilt'))


@cache
def is_dirty(cwd: str = BASEDIR) -> bool:
  origin = get_origin()
  branch = get_branch()
  if not origin or not branch:
    return True

  dirty = False
  try:
    # Actually check dirty files
    if not is_prebuilt(cwd):
      # This is needed otherwise touched files might show up as modified
      try:
        subprocess.check_call(["git", "update-index", "--refresh"], cwd=cwd)
      except subprocess.CalledProcessError:
        pass

      dirty = (subprocess.call(["git", "diff-index", "--quiet", branch, "--"], cwd=cwd)) != 0
  except subprocess.CalledProcessError:
    cloudlog.exception("git subprocess failed while checking dirty")
    dirty = True

  return dirty


@dataclass
class OpenpilotMetadata:
  version: str
  release_notes: str
  git_commit: str
  git_origin: str
  git_commit_date: str
  build_style: str
  is_dirty: bool  # whether there are local changes

  @property
  def short_version(self) -> str:
    return self.version.split('-')[0]

  @property
  def comma_remote(self) -> bool:
    # note to fork maintainers, this is used for release metrics. please do not
    # touch this to get rid of the orange startup alert. there's better ways to do that
    return self.git_normalized_origin == "github.com/commaai/openpilot"

  @property
  def git_normalized_origin(self) -> str:
    return self.git_origin \
      .replace("git@", "", 1) \
      .replace(".git", "", 1) \
      .replace("https://", "", 1) \
      .replace(":", "/", 1)


@dataclass
class BuildMetadata:
  channel: str
  openpilot: OpenpilotMetadata

  @property
  def tested_channel(self) -> bool:
    return self.channel in TESTED_BRANCHES

  @property
  def release_channel(self) -> bool:
    return self.channel in RELEASE_BRANCHES

  @property
  def canonical(self) -> str:
    return f"{self.openpilot.version}-{self.openpilot.git_commit}-{self.openpilot.build_style}"


def build_metadata_from_dict(build_metadata: dict) -> BuildMetadata:
  channel = build_metadata.get("channel", "unknown")
  openpilot_metadata = build_metadata.get("openpilot", {})
  version = openpilot_metadata.get("version", "unknown")
  release_notes = openpilot_metadata.get("release_notes", "unknown")
  git_commit = openpilot_metadata.get("git_commit", "unknown")
  git_origin = openpilot_metadata.get("git_origin", "unknown")
  git_commit_date = openpilot_metadata.get("git_commit_date", "unknown")
  build_style = openpilot_metadata.get("build_style", "unknown")
  return BuildMetadata(channel,
            OpenpilotMetadata(
              version=version,
              release_notes=release_notes,
              git_commit=git_commit,
              git_origin=git_origin,
              git_commit_date=git_commit_date,
              build_style=build_style,
              is_dirty=False))


def get_build_metadata(path: str = BASEDIR) -> BuildMetadata:
  build_metadata_path = pathlib.Path(path) / BUILD_METADATA_FILENAME

  if build_metadata_path.exists():
    build_metadata = json.loads(build_metadata_path.read_text())
    return build_metadata_from_dict(build_metadata)

  git_folder = pathlib.Path(path) / ".git"

  if git_folder.exists():
    return BuildMetadata(get_short_branch(path),
                    OpenpilotMetadata(
                      version=get_version(path),
                      release_notes=get_release_notes(path),
                      git_commit=get_commit(path),
                      git_origin=get_origin(path),
                      git_commit_date=get_commit_date(path),
                      build_style="unknown",
                      is_dirty=is_dirty(path)))

  cloudlog.exception("unable to get build metadata")
  raise Exception("invalid build metadata")


if __name__ == "__main__":
  from openpilot.common.params import Params

  params = Params()
  params.put("TermsVersion", terms_version)
  params.put("TrainingVersion", training_version)

  print(get_build_metadata())
