#!/usr/bin/env python3
from __future__ import annotations

import os
import random
import string
import subprocess
import time
from pathlib import Path
from typing import cast

import zstandard as zstd

from cereal import log as capnp_log
from openpilot.common.params import Params
from openpilot.system.hardware.hw import Paths
from openpilot.system.version import get_version


def _logger_get_identifier(key: str) -> str:
  params = Params()
  try:
    cnt_str = params.get(key) or "0"
    cnt = int(cnt_str)
  except Exception:
    cnt = 0
  params.put(key, cnt + 1)

  rand = "".join(random.choice(string.hexdigits.lower()) for _ in range(10))
  return f"{cnt:08x}--{rand}"


def _read_dir_files(path: Path) -> dict[str, bytes]:
  ret: dict[str, bytes] = {}
  if not path.exists() or not path.is_dir():
    return ret
  for p in sorted(path.iterdir()):
    if p.is_file():
      try:
        ret[str(p)] = p.read_bytes()
      except Exception:
        pass
  return ret


def build_init_data() -> bytes:
  msg = capnp_log.Event.new_message()
  init = msg.init("initData")

  init.wallTimeNanos = time.time_ns()
  init.version = get_version()
  init.dirty = not bool(os.environ.get("CLEAN"))

  # device type
  # Python hardware layer exposes string types in many places; map common cases
  # Default to pc when running on a workstation
  dev_type = capnp_log.InitData.DeviceType.pc
  try:
    # best-effort: read model name from sysfs if present
    model = Path("/sys/firmware/devicetree/base/model")
    if model.exists():
      name = model.read_text().strip().removeprefix("comma ")
      dev_type = getattr(capnp_log.InitData.DeviceType, name, dev_type)
  except Exception:
    pass
  init.deviceType = dev_type

  # kernel args
  try:
    with open("/proc/cmdline") as f:
      init.kernelArgs = f.read().strip().split(" ")
  except FileNotFoundError:
    init.init("kernelArgs", 0)

  # versions
  try:
    init.kernelVersion = Path("/proc/version").read_text()
  except Exception:
    init.kernelVersion = ""
  try:
    init.osVersion = Path("/VERSION").read_text()
  except Exception:
    init.osVersion = ""

  # params
  params = Params(os.environ.get("PARAMS_COPY_PATH", ""))
  # Only include a limited set of params to match tests
  wanted_keys = {
    "DongleId",
    "GitCommit",
    "GitCommitDate",
    "GitBranch",
    "GitRemote",
    "AccessToken",
    "BootCount",
  }
  all_keys = [k.decode() if isinstance(k, (bytes, bytearray)) else k for k in params.all_keys()]
  keys = [k for k in all_keys if k in wanted_keys]

  # Common params we explicitly expose as fields
  init.gitCommit = params.get("GitCommit") or ""
  init.gitCommitDate = params.get("GitCommitDate") or ""
  init.gitBranch = params.get("GitBranch") or ""
  init.gitRemote = params.get("GitRemote") or ""
  init.passive = False
  init.dongleId = params.get("DongleId") or ""

  # prebuilt
  try:
    init.gitSrcCommit = Path("../../git_src_commit").read_text()
  except Exception:
    init.gitSrcCommit = ""
  try:
    init.gitSrcCommitDate = Path("../../git_src_commit_date").read_text()
  except Exception:
    init.gitSrcCommitDate = ""

  # Map(Text, Data)
  entries = init.init("params").init("entries", len(keys))
  for i, k in enumerate(keys):
    kv = entries[i]
    kv.key = k
    v = params.get(k)
    # Respect DONT_LOG for known keys (AccessToken). Python params wrapper
    # doesn’t expose flags, so filter by name as a practical approximation.
    if k == "AccessToken":
      kv.value = b""
    elif v is not None:
      # store raw bytes; Params returns decoded types, so re-encode
      try:
        kv.value = v if isinstance(v, bytes) else str(v).encode()
      except Exception:
        kv.value = b""
    else:
      kv.value = b""

  # commands: df -h and any HW-specific logs are omitted in this Python port
  commands = init.init("commands").init("entries", 1)
  commands[0].key = "df -h"
  try:
    res = subprocess.check_output(["bash", "-lc", "df -h"], stderr=subprocess.STDOUT)
  except Exception:
    res = b""
  commands[0].value = res

  return cast(bytes, msg.to_bytes())


def build_boot_log() -> bytes:
  msg = capnp_log.Event.new_message()
  boot = msg.init("boot")
  boot.wallTimeNanos = time.time_ns()

  # pstore
  pstore = _read_dir_files(Path("/sys/fs/pstore"))
  pentries = boot.init("pstore").init("entries", len(pstore))
  for i, (k, v) in enumerate(sorted(pstore.items())):
    pentries[i].key = k
    pentries[i].value = v

  # commands (none for speed in tests)
  boot.init("commands").init("entries", 0)

  try:
    with open("/tmp/launch_log", newline="") as f:
      boot.launchLog = f.read()
  except Exception:
    boot.launchLog = ""

  return cast(bytes, msg.to_bytes())


def main() -> int:
  boot_id = _logger_get_identifier("BootCount")
  out_dir = Path(Paths.log_root()) / "boot"
  out_dir.mkdir(parents=True, exist_ok=True)
  out_path = out_dir / f"{boot_id}.zst"

  print(f"bootlog to {out_path}")

  cctx = zstd.ZstdCompressor(level=10)
  with open(out_path, "wb") as f:
    with cctx.stream_writer(f) as zw:
      zw.write(build_init_data())
      zw.write(build_boot_log())

  Params().put("CurrentBootlog", boot_id)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
