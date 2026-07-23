#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
  repo_root = Path(__file__).resolve().parents[3]
  sys.path[:0] = [str(repo_root), str(repo_root / "msgq_repo")]

import openpilot.cereal.messaging as messaging
from openpilot.common.hardware.hw import Paths
from openpilot.common.params import Params
from openpilot.system.loggerd.logger import build_init_data, get_identifier
from openpilot.system.loggerd.zstd_writer import ZstdFileWriter


def _read_bytes(path: str | Path) -> bytes:
  try:
    return Path(path).read_bytes()
  except OSError:
    return b""


def _read_files(path: str | Path) -> dict[str, bytes]:
  try:
    entries = list(Path(path).iterdir())
  except OSError:
    return {}
  return {entry.name: _read_bytes(entry) for entry in entries if not entry.is_dir()}


def _check_output(command: str) -> bytes:
  try:
    return subprocess.check_output(command, shell=True)
  except (OSError, subprocess.CalledProcessError):
    return b""


def build_boot_log() -> bytes:
  msg = messaging.new_message("boot", valid=True)
  boot = msg.boot
  boot.wallTimeNanos = time.time_ns()

  pstore = _read_files("/sys/fs/pstore")
  pstore_entries = boot.pstore.init("entries", len(pstore))
  for entry, (key, value) in zip(pstore_entries, sorted(pstore.items()), strict=True):
    entry.key = key
    entry.value = value

  commands = [
    '[ -x "$(command -v journalctl)" ] && journalctl -b -n 2000 -o short-monotonic --no-pager',
  ]
  command_entries = boot.commands.init("entries", len(commands))
  for entry, command in zip(command_entries, commands, strict=True):
    entry.key = command
    entry.value = _check_output(command)

  boot.launchLog = _read_bytes("/tmp/launch_log").decode("utf-8", "replace")
  return msg.to_bytes()


def main() -> None:
  identifier = get_identifier("BootCount")
  boot_dir = Path(Paths.log_root()) / "boot"
  path = boot_dir / f"{identifier}.zst"
  print(f"bootlog to {path}")

  boot_dir.mkdir(mode=0o775, parents=True, exist_ok=True)
  with ZstdFileWriter(path) as writer:
    writer.write(build_init_data())
    writer.write(build_boot_log())

  Params().put("CurrentBootlog", identifier)


if __name__ == "__main__":
  main()
