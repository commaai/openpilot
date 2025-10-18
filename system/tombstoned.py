#!/usr/bin/env python3
import datetime
import os
import re
import shutil
import signal
import subprocess
import time
import glob
from typing import NoReturn

import openpilot.system.sentry as sentry
from openpilot.system.hardware.hw import Paths
from openpilot.common.swaglog import cloudlog
from openpilot.system.version import get_build_metadata

MAX_SIZE = 1_000_000 * 100  # allow up to 100M
MAX_TOMBSTONE_FN_LEN = 62  # 85 - 23 ("<dongle id>/crash/")

TOMBSTONE_DIR = "/data/tombstones/"
APPORT_DIR = "/var/crash/"


def safe_fn(s):
  extra = ['_']
  return "".join(c for c in s if c.isalnum() or c in extra).rstrip()


def clear_apport_folder():
  for f in glob.glob(APPORT_DIR + '*'):
    try:
      os.remove(f)
    except Exception:
      pass


def get_apport_stacktrace(fn):
  try:
    cmd = f'apport-retrace -s <(cat <(echo "Package: openpilot") "{fn}")'
    return subprocess.check_output(cmd, shell=True, encoding='utf8', timeout=30, executable='/bin/bash')
  except subprocess.CalledProcessError:
    return "Error getting stacktrace"
  except subprocess.TimeoutExpired:
    return "Timeout getting stacktrace"


def get_tombstones():
  """Returns list of (filename, ctime) for all crashlogs"""
  files = []
  if os.path.exists(APPORT_DIR):
    with os.scandir(APPORT_DIR) as d:
      # Loop over first 1000 directory entries
      for _, f in zip(range(1000), d, strict=False):
        if f.name.startswith("tombstone"):
          files.append((f.path, int(f.stat().st_ctime)))
        elif f.name.endswith(".crash") and f.stat().st_mode == 0o100640:
          files.append((f.path, int(f.stat().st_ctime)))
  return files


def report_tombstone_apport(fn):
  f_size = os.path.getsize(fn)
  if f_size > MAX_SIZE:
    cloudlog.error(f"Tombstone {fn} too big, {f_size}. Skipping...")
    return

  message = ""  # One line description of the crash
  contents = ""  # Full file contents without coredump
  path = ""  # File path relative to openpilot directory

  proc_maps = False

  with open(fn) as f:
    for line in f:
      if "CoreDump" in line:
        break
      elif "ProcMaps" in line:
        proc_maps = True
      elif "ProcStatus" in line:
        proc_maps = False

      if not proc_maps:
        contents += line

      if "ExecutablePath" in line:
        path = line.strip().split(': ')[-1]
        path = path.replace('/data/openpilot/', '')
        message += path
      elif "Signal" in line:
        message += " - " + line.strip()

        try:
          sig_num = int(line.strip().split(': ')[-1])
          message += " (" + signal.Signals(sig_num).name + ")"
        except ValueError:
          pass

  stacktrace = get_apport_stacktrace(fn)
  stacktrace_s = stacktrace.split('\n')
  crash_function = "No stacktrace"

  if len(stacktrace_s) > 2:
    found = False

    # Try to find first entry in openpilot, fall back to first line
    for line in stacktrace_s:
      if "at selfdrive/" in line:
        crash_function = line
        found = True
        break

    if not found:
      crash_function = stacktrace_s[1]

    # Remove arguments that can contain pointers to make sentry one-liner unique
    crash_function = " ".join(x for x in crash_function.split(' ')[1:] if not x.startswith('0x'))
    crash_function = re.sub(r'\(.*?\)', '', crash_function)

  contents = stacktrace + "\n\n" + contents
  message = message + " - " + crash_function
  sentry.report_tombstone(fn, message, contents)

  # Copy crashlog to upload folder
  clean_path = path.replace('/', '_')
  date = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

  build_metadata = get_build_metadata()

  new_fn = f"{date}_{(build_metadata.openpilot.git_commit or 'nocommit')[:8]}_{safe_fn(clean_path)}"[:MAX_TOMBSTONE_FN_LEN]

  crashlog_dir = os.path.join(Paths.log_root(), "crash")
  os.makedirs(crashlog_dir, exist_ok=True)

  # Files could be on different filesystems, copy, then delete
  shutil.copy(fn, os.path.join(crashlog_dir, new_fn))

  try:
    os.remove(fn)
  except PermissionError:
    pass


def main() -> NoReturn:
  should_report = sentry.init(sentry.SentryProject.SELFDRIVE_NATIVE)

  # Clear apport folder on start, otherwise duplicate crashes won't register
  clear_apport_folder()
  initial_tombstones = set(get_tombstones())

  while True:
    now_tombstones = set(get_tombstones())

    for fn, _ in (now_tombstones - initial_tombstones):
      # clear logs if we're not interested in them
      if not should_report:
        try:
          os.remove(fn)
        except Exception:
          pass
        continue

      try:
        cloudlog.info(f"reporting new tombstone {fn}")
        if fn.endswith(".crash"):
          report_tombstone_apport(fn)
        else:
          cloudlog.error(f"unknown crash type: {fn}")
      except Exception:
        cloudlog.exception(f"Error reporting tombstone {fn}")

    initial_tombstones = now_tombstones
    time.sleep(5)


if __name__ == "__main__":
  main()
