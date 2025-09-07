#!/usr/bin/env python3
import os
import random
import re
import subprocess
import time
from pathlib import Path

import zstandard as zstd
from cereal import log
from openpilot.common.basedir import BASEDIR
from openpilot.common.file_helpers import LOG_COMPRESSION_LEVEL
from openpilot.common.params import Params
from openpilot.system.hardware import HARDWARE
from openpilot.system.hardware.hw import Paths


def read_file(path: str) -> bytes:
  try:
    with open(path, "rb") as f:
      return f.read()
  except Exception:
    return b""


def read_files_in_dir(path: str) -> dict[str, bytes]:
  files: dict[str, bytes] = {}
  try:
    for fn in os.listdir(path):
      fp = os.path.join(path, fn)
      if os.path.isfile(fp):
        files[fn] = read_file(fp)
  except Exception:
    pass
  return files


def check_output(cmd: str) -> bytes:
  try:
    return subprocess.check_output(cmd, shell=True)
  except subprocess.CalledProcessError as e:
    return e.output


def get_version() -> str:
  vh = Path(BASEDIR) / "common" / "version.h"
  try:
    m = re.search(r'COMMA_VERSION "([^"]+)"', vh.read_text())
    return m.group(1) if m else ""
  except Exception:
    return ""


def logger_get_identifier(key: str) -> str:
  params = Params()
  try:
    cnt = int(params.get(key) or 0)
  except Exception:
    cnt = 0
  params.put(key, str(cnt + 1))
  rand = ''.join(random.choice('0123456789abcdef') for _ in range(10))
  return f"{cnt:08x}--{rand}"


def logger_build_init_data() -> bytes:
  msg = log.Event.new_message()
  init = msg.init('initData')
  init.wallTimeNanos = time.time_ns()
  init.version = get_version()
  init.dirty = not os.environ.get("CLEAN")
  init.deviceType = getattr(log.InitData.DeviceType, HARDWARE.get_device_type())

  try:
    kernel_args = read_file('/proc/cmdline').decode().split()
  except Exception:
    kernel_args = []
  ka = init.init('kernelArgs', len(kernel_args))
  for i, a in enumerate(kernel_args):
    ka[i] = a

  init.kernelVersion = read_file('/proc/version').decode(errors='ignore')
  init.osVersion = read_file('/VERSION').decode(errors='ignore')

  params = Params(os.getenv('PARAMS_COPY_PATH', ''))
  param_root = params.get_param_path()
  params_map: dict[str, bytes] = {}
  for k in params.all_keys():
    try:
      with open(os.path.join(param_root, k), 'rb') as f:
        params_map[k] = f.read()
    except Exception:
      pass

  init.gitCommit = params_map.get('GitCommit', b'').decode(errors='ignore')
  init.gitCommitDate = params_map.get('GitCommitDate', b'').decode(errors='ignore')
  init.gitBranch = params_map.get('GitBranch', b'').decode(errors='ignore')
  init.gitRemote = params_map.get('GitRemote', b'').decode(errors='ignore')
  init.passive = False
  init.dongleId = params_map.get('DongleId', b'').decode(errors='ignore')
  init.gitSrcCommit = read_file('../../git_src_commit').decode(errors='ignore')
  init.gitSrcCommitDate = read_file('../../git_src_commit_date').decode(errors='ignore')

  lparams = init.init('params').init('entries', len(params_map))
  for i, (key, value) in enumerate(params_map.items()):
    lparams[i].key = key
    lparams[i].value = value

  log_commands = ['df -h']
  commands = init.init('commands').init('entries', len(log_commands))
  for i, cmd in enumerate(log_commands):
    commands[i].key = cmd
    commands[i].value = check_output(cmd)

  return msg.to_bytes()


def build_boot_log() -> bytes:
  msg = log.Event.new_message()
  boot = msg.init('boot')
  boot.wallTimeNanos = time.time_ns()

  pstore_map = read_files_in_dir('/sys/fs/pstore')
  lpstore = boot.init('pstore').init('entries', len(pstore_map))
  for i, (k, v) in enumerate(pstore_map.items()):
    lpstore[i].key = k
    lpstore[i].value = v

  bootlog_commands = ['[ -x "$(command -v journalctl)" ] && journalctl -o short-monotonic']
  commands = boot.init('commands').init('entries', len(bootlog_commands))
  for i, cmd in enumerate(bootlog_commands):
    commands[i].key = cmd
    commands[i].value = check_output(cmd)

  boot.launchLog = read_file('/tmp/launch_log').decode(errors='ignore')
  return msg.to_bytes()


def main() -> None:
  log_id = logger_get_identifier('BootCount')
  path = os.path.join(Paths.log_root(), 'boot', f'{log_id}.zst')
  print(f'bootlog to {path}')
  os.makedirs(os.path.dirname(path), mode=0o775, exist_ok=True)

  cctx = zstd.ZstdCompressor(level=LOG_COMPRESSION_LEVEL)
  with open(path, 'wb') as f:
    with cctx.stream_writer(f) as writer:
      writer.write(logger_build_init_data())
      writer.write(build_boot_log())

  Params().put('CurrentBootlog', log_id)


if __name__ == '__main__':
  main()
