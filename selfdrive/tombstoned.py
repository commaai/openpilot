#!/usr/bin/env python3
import re
import os
import time
import subprocess

from raven import Client
from raven.transport.http import HTTPTransport

from selfdrive.hardware import TICI
from selfdrive.swaglog import cloudlog
from selfdrive.version import version, origin, branch, dirty

MAX_SIZE = 100000 * 10  # Normal size is 40-100k, allow up to 1M
if TICI:
  MAX_SIZE = MAX_SIZE * 100  # Allow larger size for tici


def get_apport_stacktrace(fn):
  try:
    return subprocess.check_output(f'apport-retrace -s <(cat <(echo "Package: openpilot") "{fn}")', shell=True, encoding='utf8', timeout=30)
  except subprocess.CalledProcessError:
    return "Error getting stacktrace"
  except subprocess.TimeoutExpired:
    return "Timeout getting stacktrace"


def get_tombstones():
  """Returns list of (filename, ctime) for all tombstones in /data/tombstones
  and apport crashlogs in /var/crash"""
  files = []
  for folder in ["/data/tombstones/", "/var/crash/"]:
    if os.path.exists(folder):
      with os.scandir(folder) as d:

        # Loop over first 1000 directory entries
        for _, f in zip(range(1000), d):
          if f.name.startswith("tombstone"):
            files.append((f.path, int(f.stat().st_ctime)))
          elif f.name.endswith(".crash") and f.stat().st_mode == 0o100640:
            files.append((f.path, int(f.stat().st_ctime)))
  return files


def report_tombstone_android(fn, client):
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

  # Cut off fault addr
  fault_idx = message.find(', fault addr')
  if fault_idx >= 0:
    message = message[:fault_idx]

  cloudlog.error({'tombstone': message})
  client.captureMessage(
    message=message,
    sdk={'name': 'tombstoned', 'version': '0'},
    extra={
      'tombstone_fn': fn,
      'tombstone': contents
    },
  )


def report_tombstone_apport(fn, client):
  f_size = os.path.getsize(fn)
  if f_size > MAX_SIZE:
    cloudlog.error(f"Tombstone {fn} too big, {f_size}. Skipping...")
    return

  include_in_messge = ["ProblemType", "ExecutablePath", "Signal"]
  message = ""  # One line description of the crash
  contents = ""  # Full file contents without coredump

  with open(fn) as f:
    for line in f:
      if "CoreDump" in line:
        break

      contents += line
      if any([x in line for x in include_in_messge]):
        message += line.strip() + " "

  # Get stacktrace using gdb
  stacktrace = get_apport_stacktrace(fn)
  contents += stacktrace + "\n" + contents

  # Get function and line of crash, but remove function arguments as they are not static
  crash_function = stacktrace.split('\n')[1][3:]
  crash_function = re.sub(r'\(.*?\)', '', crash_function)

  message = message + " " + crash_function

  cloudlog.error({'tombstone': message})
  client.captureMessage(
    message=message,
    sdk={'name': 'tombstoned', 'version': '0'},
    extra={
      'tombstone_fn': fn,
      'tombstone': contents
    },
  )

  # TODO: move file to /data/media to be uploaded


def main():
  initial_tombstones = set(get_tombstones())

  tags = {
    'dirty': dirty,
    'origin': origin,
    'branch': branch
  }
  client = Client('https://d3b175702f62402c91ade04d1c547e68:b20d68c813c74f63a7cdf9c4039d8f56@sentry.io/157615',
                  install_sys_hook=False, transport=HTTPTransport, release=version, tags=tags, string_max_length=10000)

  client.user_context({'id': os.environ.get('DONGLE_ID')})
  while True:
    now_tombstones = set(get_tombstones())

    for fn, _ in (now_tombstones - initial_tombstones):
      try:
        cloudlog.info(f"reporting new tombstone {fn}")
        if fn.endswith(".crash"):
          report_tombstone_apport(fn, client)
        else:
          report_tombstone_android(fn, client)
      except Exception:
        cloudlog.exception(f"Error reporting tombstone {fn}")

    initial_tombstones = now_tombstones
    time.sleep(5)


if __name__ == "__main__":
  main()
