import subprocess
from pathlib import Path


def read_file(path: str | Path, default: bytes) -> bytes:
  try:
    return Path(path).read_bytes()
  except OSError:
    return default


def read_text(path: str | Path, default: str) -> str:
  try:
    return Path(path).read_text()
  except OSError:
    return default


def check_output(command: str | list[str], default: bytes, *, shell: bool = False, cwd: str | Path | None = None) -> bytes:
  try:
    return subprocess.check_output(command, shell=shell, cwd=cwd)
  except (OSError, subprocess.CalledProcessError):
    return default
