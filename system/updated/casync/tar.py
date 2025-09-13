import pathlib
import tarfile
from typing import IO
from collections.abc import Callable


def include_default(_) -> bool:
  return True


def create_tar_archive(filename: pathlib.Path, directory: pathlib.Path, include: Callable[[pathlib.Path], bool] = include_default):
  """Creates a tar archive of a directory"""

  with tarfile.open(filename, 'w') as tar:
    for file in sorted(directory.rglob("*"), key=lambda f: f.stat().st_size if f.is_file() else 0, reverse=True):
      if not include(file):
        continue
      relative_path = str(file.relative_to(directory))
      if file.is_symlink():
        info = tarfile.TarInfo(relative_path)
        info.type = tarfile.SYMTYPE
        info.linkpath = str(file.readlink())
        tar.addfile(info)

      elif file.is_file():
        info = tarfile.TarInfo(relative_path)
        info.size = file.stat().st_size
        info.type = tarfile.REGTYPE
        info.mode = file.stat().st_mode
        with file.open('rb') as f:
          tar.addfile(info, f)


def extract_tar_archive(fh: IO[bytes], directory: pathlib.Path):
  """Extracts a tar archive to a directory"""

  tar = tarfile.open(fileobj=fh, mode='r')
  tar.extractall(str(directory), filter=lambda info, path: info)
  tar.close()
