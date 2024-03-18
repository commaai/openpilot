import json
import pathlib
import subprocess

from openpilot.system.version import get_release_notes, get_version


CASYNC_ARGS = ["--with=symlinks", "--with=permissions"]
CHANNEL_MANIFEST_FILE = "channel.json" # file that contains details of the current release
CASYNC_FILES = [CHANNEL_MANIFEST_FILE, ".caexclude"]


def run(cmd):
  return subprocess.check_output(cmd)


def get_exclude_set(path) -> set[str]:
  exclude_set = set(CASYNC_FILES)

  for file in path.rglob("*"):
    if file.is_file() or file.is_symlink():

      while file.resolve() != path.resolve():
        exclude_set.add(str(file.relative_to(path)))

        file = file.parent

  return exclude_set


def create_caexclude_file(path: pathlib.Path):
  with open(path / ".caexclude", "w") as f:
    # exclude everything except the paths already in the release
    f.write("*\n")
    f.write(".*\n")

    for file in sorted(get_exclude_set(path)):
      f.write(f"!{file}\n")


def create_manifest(channel, version, release_notes):
  return {
    "name": channel,
    "openpilot": {
      "version": version,
      "release_notes": release_notes
    }
  }

def create_manifest_file(path: pathlib.Path, channel: str):
  version = get_version(str(path))
  release_notes = get_release_notes(str(path))
  with open(path / CHANNEL_MANIFEST_FILE, "w") as f:
    f.write(json.dumps(create_manifest(channel, version, release_notes)))


def create_casync_channel(target_dir: pathlib.Path, output_dir: pathlib.Path, channel: str):
  caidx_file = output_dir / f"{channel}.caidx"
  run(["casync", "make", *CASYNC_ARGS, caidx_file, target_dir])
  digest = run(["casync", "digest", *CASYNC_ARGS, target_dir]).decode("utf-8").strip()
  return digest, caidx_file
