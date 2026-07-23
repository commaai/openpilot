#!/usr/bin/env python3

import time
from pathlib import Path

import openpilot.cereal.messaging as messaging
import zstandard as zstd
from openpilot.common.hardware.hw import Paths
from openpilot.common.params import Params
from openpilot.common.utils import LOG_COMPRESSION_LEVEL
from openpilot.system.loggerd.logger import build_init_data, check_output, get_identifier, read_file


def _read_files(path: str | Path) -> dict[str, bytes]:
  try:
    entries = list(Path(path).iterdir())
  except OSError:
    return {}
  return {entry.name: read_file(entry) for entry in entries if not entry.is_dir()}


def build_boot_log() -> bytes:
  msg = messaging.new_message("boot", valid=True)
  boot = msg.boot
  boot.wallTimeNanos = time.time_ns()

  pstore = _read_files("/sys/fs/pstore")
  pstore_entries = boot.pstore.init("entries", len(pstore))
  for entry, (key, value) in zip(pstore_entries, sorted(pstore.items()), strict=True):
    entry.key = key
    entry.value = value

  command = '[ -x "$(command -v journalctl)" ] && journalctl -b -n 2000 -o short-monotonic --no-pager'
  command_entry = boot.commands.init("entries", 1)[0]
  command_entry.key = command
  command_entry.value = check_output(command)

  boot.launchLog = read_file("/tmp/launch_log").decode("utf-8", "replace")
  return msg.to_bytes()


def create_bootlog(params_path: str = "") -> Path:
  identifier = get_identifier("BootCount")
  boot_dir = Path(Paths.log_root()) / "boot"
  path = boot_dir / f"{identifier}.zst"
  print(f"bootlog to {path}")

  boot_dir.mkdir(mode=0o775, parents=True, exist_ok=True)
  with zstd.open(path, "wb", cctx=zstd.ZstdCompressor(level=LOG_COMPRESSION_LEVEL)) as writer:
    writer.write(build_init_data(params_path))
    writer.write(build_boot_log())

  Params().put("CurrentBootlog", identifier, block=True)
  return path


if __name__ == "__main__":
  create_bootlog()
