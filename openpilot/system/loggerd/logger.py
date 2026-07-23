import os
import secrets
import subprocess
import time
from pathlib import Path

import openpilot.cereal.messaging as messaging
from openpilot.common.basedir import BASEDIR
from openpilot.common.hardware import HARDWARE, TICI
from openpilot.common.params import ParamKeyFlag, Params
from openpilot.common.version import get_version


def read_file(path: str | Path) -> bytes:
  try:
    return Path(path).read_bytes()
  except OSError:
    return b""


def _read_text(path: str | Path) -> str:
  return read_file(path).decode("utf-8", "replace")


def check_output(command: str) -> bytes:
  try:
    return subprocess.check_output(command, shell=True)
  except (OSError, subprocess.CalledProcessError):
    return b""


def _hardware_init_logs() -> dict[str, bytes]:
  if not TICI:
    return {}

  logs = {
    "/BUILD": read_file("/BUILD"),
    "lsblk": check_output("lsblk -o NAME,SIZE,STATE,VENDOR,MODEL,REV,SERIAL"),
    "SOM ID": read_file("/sys/devices/platform/vendor/vendor:gpio-som-id/som_id"),
  }

  boot_slot = check_output("abctl --boot_slot")
  logs["boot slot"] = boot_slot.split(b"\n", 1)[0]
  logs["boot temp"] = read_file("/dev/disk/by-partlabel/ssd").rstrip(b"\0\r\n")

  for part in ("xbl", "abl", "aop", "devcfg", "xbl_config"):
    for slot in ("a", "b"):
      partition = f"{part}_{slot}"
      logs[partition] = check_output(f"sha256sum /dev/disk/by-partlabel/{partition}").split(b" ", 1)[0]

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


def build_init_data(params_path: str = "") -> bytes:
  msg = messaging.new_message("initData", valid=True)
  init = msg.initData

  init.wallTimeNanos = time.time_ns()
  init.version = get_version()
  init.dirty = os.getenv("CLEAN") is None
  init.deviceType = HARDWARE.get_device_type()

  init.kernelArgs = _read_text("/proc/cmdline").split()
  init.kernelVersion = _read_text("/proc/version")
  init.osVersion = _read_text("/VERSION")

  params = Params(params_path)
  params_map = _raw_params(params)
  init.gitCommit = params_map.get("GitCommit", b"").decode("utf-8", "replace")
  init.gitCommitDate = params_map.get("GitCommitDate", b"").decode("utf-8", "replace")
  init.gitBranch = params_map.get("GitBranch", b"").decode("utf-8", "replace")
  init.gitRemote = params_map.get("GitRemote", b"").decode("utf-8", "replace")
  init.passive = False
  init.dongleId = params_map.get("DongleId", b"").decode("utf-8", "replace")

  init.gitSrcCommit = _read_text(Path(BASEDIR) / "openpilot" / "git_src_commit")
  init.gitSrcCommitDate = _read_text(Path(BASEDIR) / "openpilot" / "git_src_commit_date")

  param_entries = init.params.init("entries", len(params_map))
  for entry, (key, value) in zip(param_entries, sorted(params_map.items()), strict=True):
    entry.key = key
    entry.value = b"" if params.get_flag(key) & ParamKeyFlag.DONT_LOG else value

  commands = {"df -h": check_output("df -h"), **dict(sorted(_hardware_init_logs().items()))}
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
  params.put(key, count + 1, block=True)
  return f"{count:08x}--{secrets.token_hex(5)}"
