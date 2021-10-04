#!/usr/bin/env python3
import datetime
import os
import re
import shutil
import signal
import subprocess
import time
import glob

import sentry_sdk

from common.params import Params
from common.file_helpers import mkdirs_exists_ok
from selfdrive.hardware import TICI, HARDWARE
from selfdrive.loggerd.config import ROOT
from selfdrive.swaglog import cloudlog
from selfdrive.version import branch, commit, dirty, origin, version

MAX_SIZE = 100000 * 10  # mal size is 40-100k, allow up to 1M
if TICI:
  MAX_SIZE = MAX_SIZE * 100  # Allow larger size for tici since files contain coredump
MAX_TOMBSTONE_FN_LEN = 62  # 85 - 23 ("<dongle id>/crash/")

TOMBSTONE_DIR = "/data/tombstones/"
APPORT_DIR = "/var/crash/"


def safe_fn(s):
  extra = ['_']
  return "".join(c for c in s if c.isalnum() or c in extra).rstrip()


def sentry_report(fn, message, contents):
  cloudlog.error({'tombstone': message})

  with sentry_sdk.configure_scope() as scope:
      scope.set_extra("tombstone_fn", fn)
      scope.set_extra("tombstone", contents)
      sentry_sdk.capture_message(message=message)
      sentry_sdk.flush()


def clear_apport_folder():
  for f in glob.glob(f"{APPORT_DIR}*"):
    try:
      os.remove(f)
    except Exception:
      pass


def get_apport_stacktrace(fn):
  try:
    cmd = f'apport-retrace -s <(cat <(echo "Package: openpilot") "{fn}")'
    return subprocess.check_output(cmd, shell=True, encoding='utf8', timeout=30, executable='/bin/bash')  # pylint: disable=unexpected-keyword-arg
  except subprocess.CalledProcessError:
    return "Error getting stacktrace"
  except subprocess.TimeoutExpired:
    return "Timeout getting stacktrace"


def get_tombstones():
  """Returns list of (filename, ctime) for all tombstones in /data/tombstones
  and apport crashlogs in /var/crash"""
  files = []
  for folder in [TOMBSTONE_DIR, APPORT_DIR]:
    if os.path.exists(folder):
      with os.scandir(folder) as d:

        # Loop over first 1000 directory entries
        for _, f in zip(range(1000), d):
          if f.name.startswith("tombstone"):
            files.append((f.path, int(f.stat().st_ctime)))
          elif f.name.endswith(".crash") and f.stat().st_mode == 0o100640:
            files.append((f.path, int(f.stat().st_ctime)))
  return files


def report_tombstone_android(fn):
  f_size = os.path.getsize(fn)
  if f_size > MAX_SIZE:
    cloudlog.error(f"Tombstone {fn} too big, {f_size}. Skipping...")
    return

  with open(fn, encoding='ISO-8859-1') as f:
    contents = f.read()

  message = " ".join(contents.split('\n')[5:7])

  # Cut off pid/tid, since that varies per run
  name_idx = message.find('name')
  if name_idx >= 0:
    message = message[name_idx:]

  executable = ""
  start_exe_idx = message.find('>>> ')
  end_exe_idx = message.find(' <<<')
  if start_exe_idx >= 0 and end_exe_idx >= 0:
    executable = message[start_exe_idx + 4:end_exe_idx]

  # Cut off fault addr
  fault_idx = message.find(', fault addr')
  if fault_idx >= 0:
    message = message[:fault_idx]

  sentry_report(fn, message, contents)

  # Copy crashlog to upload folder
  clean_path = executable.replace('./', '').replace('/', '_')
  date = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

  new_fn = f"{date}_{commit[:8]}_{safe_fn(clean_path)}"[:MAX_TOMBSTONE_FN_LEN]

  crashlog_dir = os.path.join(ROOT, "crash")
  mkdirs_exists_ok(crashlog_dir)

  shutil.copy(fn, os.path.join(crashlog_dir, new_fn))


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
        message += f" - {line.strip()}"

        try:
          sig_num = int(line.strip().split(': ')[-1])
          message += f" ({signal.Signals(sig_num).name})"  # pylint: disable=no-member
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

  contents = f"{stacktrace}\n\n{contents}"
  message = f"{message} - {crash_function}"
  sentry_report(fn, message, contents)

  # Copy crashlog to upload folder
  clean_path = path.replace('/', '_')
  date = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

  new_fn = f"{date}_{commit[:8]}_{safe_fn(clean_path)}"[:MAX_TOMBSTONE_FN_LEN]

  crashlog_dir = os.path.join(ROOT, "crash")
  mkdirs_exists_ok(crashlog_dir)

  # Files could be on different filesystems, copy, then delete
  shutil.copy(fn, os.path.join(crashlog_dir, new_fn))

  try:
    os.remove(fn)
  except PermissionError:
    pass


def main():
  clear_apport_folder()  # Clear apport folder on start, otherwise duplicate crashes won't register
  initial_tombstones = set(get_tombstones())

  sentry_sdk.utils.MAX_STRING_LENGTH = 8192
  sentry_sdk.init("https://a40f22e13cbc4261873333c125fc9d38@o33823.ingest.sentry.io/157615",
                  default_integrations=False, release=version)

  dongle_id = Params().get("DongleId", encoding='utf-8')
  sentry_sdk.set_user({"id": dongle_id})
  sentry_sdk.set_tag("dirty", dirty)
  sentry_sdk.set_tag("origin", origin)
  sentry_sdk.set_tag("branch", branch)
  sentry_sdk.set_tag("commit", commit)
  sentry_sdk.set_tag("device", HARDWARE.get_device_type())

  while True:
    now_tombstones = set(get_tombstones())

    for fn, _ in (now_tombstones - initial_tombstones):
      try:
        cloudlog.info(f"reporting new tombstone {fn}")
        if fn.endswith(".crash"):
          report_tombstone_apport(fn)
        else:
          report_tombstone_android(fn)
      except Exception:
        cloudlog.exception(f"Error reporting tombstone {fn}")

    initial_tombstones = now_tombstones
    time.sleep(5)


if __name__ == "__main__":
  main()
