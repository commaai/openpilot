import subprocess
from pathlib import Path


def read_file(path: str | Path, default: bytes) -> bytes:
  try:
    return Path(path).read_bytes()
  except OSError:
    return default


def check_output(command: str | list[str], default: bytes, *, shell: bool = False) -> bytes:
  try:
    return subprocess.check_output(command, shell=shell)
  except (OSError, subprocess.CalledProcessError):
    return default
