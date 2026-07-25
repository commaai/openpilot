import json
import subprocess
from pathlib import Path
from typing import Any


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


def read_int(path: str | Path, default: int = 0, *, base: int = 10) -> int:
  try:
    return int(read_text(path, ""), base)
  except ValueError:
    return default


def read_json(path: str | Path, default: Any) -> Any:
  try:
    return json.loads(Path(path).read_text())
  except (OSError, UnicodeError, json.JSONDecodeError):
    return default


def check_output(command: str | list[str], default: bytes, *, shell: bool = False, cwd: str | Path | None = None) -> bytes:
  try:
    return subprocess.check_output(command, shell=shell, cwd=cwd)
  except (OSError, subprocess.CalledProcessError):
    return default
