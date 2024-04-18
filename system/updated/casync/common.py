import dataclasses
import json
import pathlib
import subprocess

from openpilot.system.version import BUILD_METADATA_FILENAME, BuildMetadata
from openpilot.system.updated.casync import tar


CASYNC_ARGS = ["--with=symlinks", "--with=permissions", "--compression=xz", "--chunk-size=16M"]
CASYNC_FILES = [BUILD_METADATA_FILENAME]


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


def create_build_metadata_file(path: pathlib.Path, build_metadata: BuildMetadata):
  with open(path / BUILD_METADATA_FILENAME, "w") as f:
    build_metadata_dict = dataclasses.asdict(build_metadata)
    build_metadata_dict["openpilot"].pop("is_dirty")  # this is determined at runtime
    build_metadata_dict.pop("channel")                # channel is unrelated to the build itself
    f.write(json.dumps(build_metadata_dict))


def is_not_git(path: pathlib.Path) -> bool:
  return ".git" not in path.parts


def create_casync_tar_package(target_dir: pathlib.Path, output_path: pathlib.Path):
  tar.create_tar_archive(output_path, target_dir, is_not_git)


def create_casync_from_file(file: pathlib.Path, output_dir: pathlib.Path, caibx_name: str):
  caibx_file = output_dir / f"{caibx_name}.caibx"
  run(["casync", "make", *CASYNC_ARGS, caibx_file, str(file)])

  return caibx_file


def create_casync_release(target_dir: pathlib.Path, output_dir: pathlib.Path, caibx_name: str):
  tar_file = output_dir / f"{caibx_name}.tar"
  create_casync_tar_package(target_dir, tar_file)
  caibx_file = create_casync_from_file(tar_file, output_dir, caibx_name)
  tar_file.unlink()
  digest = run(["casync", "digest", *CASYNC_ARGS, target_dir]).decode("utf-8").strip()
  return digest, caibx_file
