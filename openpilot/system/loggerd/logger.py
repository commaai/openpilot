from __future__ import annotations

import os
import re
import secrets
import subprocess
import time
from pathlib import Path

from openpilot.cereal import log
import openpilot.cereal.messaging as messaging
from openpilot.common.basedir import BASEDIR
from openpilot.common.hardware import HARDWARE, TICI
from openpilot.common.hardware.hw import Paths
from openpilot.common.params import Params
from openpilot.common.version import get_version
from openpilot.system.loggerd.zstd_writer import ZstdFileWriter


SentinelType = log.Sentinel.SentinelType
PARAMS_KEYS_PATH = Path(BASEDIR) / "openpilot/common/params_keys.h"


def _read_bytes(path: str | Path) -> bytes:
  try:
    return Path(path).read_bytes()
  except OSError:
    return b""


def _read_text(path: str | Path) -> str:
  return _read_bytes(path).decode("utf-8", "replace")


def _check_output(command: str) -> bytes:
  try:
    return subprocess.check_output(command, shell=True)
  except (OSError, subprocess.CalledProcessError):
    return b""


def _hardware_init_logs() -> dict[str, bytes]:
  if not TICI:
    return {}

  logs = {
    "/BUILD": _read_bytes("/BUILD"),
    "lsblk": _check_output("lsblk -o NAME,SIZE,STATE,VENDOR,MODEL,REV,SERIAL"),
    "SOM ID": _read_bytes("/sys/devices/platform/vendor/vendor:gpio-som-id/som_id"),
  }

  boot_slot = _check_output("abctl --boot_slot")
  logs["boot slot"] = boot_slot.split(b"\n", 1)[0]
  logs["boot temp"] = _read_bytes("/dev/disk/by-partlabel/ssd").rstrip(b"\0\r\n")

  for part in ("xbl", "abl", "aop", "devcfg", "xbl_config"):
    for slot in ("a", "b"):
      partition = f"{part}_{slot}"
      logs[partition] = _check_output(f"sha256sum /dev/disk/by-partlabel/{partition}").split(b" ", 1)[0]

  return logs


def _raw_params(params: Params) -> dict[str, bytes]:
  values = {}
  try:
    entries = list(Path(params.get_param_path()).iterdir())
  except OSError:
    return values

  for entry in entries:
    if entry.is_dir():
      continue
    try:
      value = entry.read_bytes()
    except OSError:
      continue
    values[entry.name] = value
  return values


def _dont_log_param_keys() -> set[str]:
  definitions = _read_text(PARAMS_KEYS_PATH)
  return set(re.findall(r'\{"([^"]+)", \{[^}\n]*\bDONT_LOG\b', definitions))


def build_init_data() -> bytes:
  msg = messaging.new_message("initData", valid=True)
  init = msg.initData

  init.wallTimeNanos = time.time_ns()
  init.version = get_version()
  init.dirty = os.getenv("CLEAN") is None
  init.deviceType = HARDWARE.get_device_type()

  init.kernelArgs = _read_text("/proc/cmdline").split()
  init.kernelVersion = _read_text("/proc/version")
  init.osVersion = _read_text("/VERSION")

  params = Params(os.getenv("PARAMS_COPY_PATH", ""))
  params_map = _raw_params(params)
  init.gitCommit = params_map.get("GitCommit", b"").decode("utf-8", "replace")
  init.gitCommitDate = params_map.get("GitCommitDate", b"").decode("utf-8", "replace")
  init.gitBranch = params_map.get("GitBranch", b"").decode("utf-8", "replace")
  init.gitRemote = params_map.get("GitRemote", b"").decode("utf-8", "replace")
  init.passive = False
  init.dongleId = params_map.get("DongleId", b"").decode("utf-8", "replace")

  init.gitSrcCommit = _read_text(Path(BASEDIR) / "openpilot" / "git_src_commit")
  init.gitSrcCommitDate = _read_text(Path(BASEDIR) / "openpilot" / "git_src_commit_date")

  dont_log_keys = _dont_log_param_keys()
  param_entries = init.params.init("entries", len(params_map))
  for entry, (key, value) in zip(param_entries, sorted(params_map.items()), strict=True):
    entry.key = key
    entry.value = b"" if key in dont_log_keys else value

  commands = {"df -h": _check_output("df -h"), **dict(sorted(_hardware_init_logs().items()))}
  command_entries = init.commands.init("entries", len(commands))
  for entry, (key, value) in zip(command_entries, commands.items(), strict=True):
    entry.key = key
    entry.value = value

  return msg.to_bytes()


def get_identifier(key: str) -> str:
  params = Params()
  try:
    count = int(params.get(key) or 0)
  except (TypeError, ValueError):
    count = 0
  params.put(key, count + 1)
  return f"{count:08x}--{secrets.token_hex(5)}"


def sentinel_message(sentinel_type, signal: int = 0) -> bytes:
  msg = messaging.new_message("sentinel", valid=True)
  msg.sentinel.type = sentinel_type
  msg.sentinel.signal = signal
  return msg.to_bytes()


class LoggerState:
  def __init__(self, log_root: str | Path | None = None):
    self.route_name = get_identifier("RouteCount")
    self.route_path = Path(log_root or Paths.log_root()) / self.route_name
    self.init_data = build_init_data()
    self.part = -1
    self.exit_signal = 0
    self.segment_path: Path | None = None
    self.lock_file: Path | None = None
    self.rlog: ZstdFileWriter | None = None
    self.qlog: ZstdFileWriter | None = None
    self.closed = False

  @property
  def segment(self) -> int:
    return self.part

  def write(self, data: bytes, in_qlog: bool) -> None:
    assert self.rlog is not None and self.qlog is not None
    self.rlog.write(data)
    if in_qlog:
      self.qlog.write(data)

  def _close_logs(self) -> None:
    if self.rlog is not None:
      self.rlog.close()
      self.rlog = None
    if self.qlog is not None:
      self.qlog.close()
      self.qlog = None

  def next(self) -> None:
    if self.rlog is not None:
      self.write(sentinel_message(SentinelType.endOfSegment), True)
      self._close_logs()
      if self.lock_file is not None:
        self.lock_file.unlink(missing_ok=True)

    self.part += 1
    self.segment_path = Path(f"{self.route_path}--{self.part}")
    self.segment_path.mkdir(mode=0o775, parents=True)
    self.lock_file = self.segment_path / "rlog.lock"
    self.lock_file.touch()

    self.rlog = ZstdFileWriter(self.segment_path / "rlog.zst")
    self.qlog = ZstdFileWriter(self.segment_path / "qlog.zst")
    self.write(self.init_data, True)
    start_type = SentinelType.startOfSegment if self.part > 0 else SentinelType.startOfRoute
    self.write(sentinel_message(start_type), True)

  def close(self) -> None:
    if self.closed:
      return

    if self.rlog is not None:
      self.write(sentinel_message(SentinelType.endOfRoute, self.exit_signal), True)
      self._close_logs()
    if self.lock_file is not None:
      self.lock_file.unlink(missing_ok=True)
    self.closed = True

  def __enter__(self) -> LoggerState:
    return self

  def __exit__(self, exc_type, exc, traceback) -> None:
    self.close()
