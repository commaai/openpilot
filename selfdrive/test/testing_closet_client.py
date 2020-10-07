#!/usr/bin/env python3
import errno
import fcntl
import os
import signal
import subprocess
import sys
import time

import requests

from common.params import Params
from common.timeout import Timeout

HOST = "testing.comma.life"


def unblock_stdout():
  # get a non-blocking stdout
  child_pid, child_pty = os.forkpty()
  if child_pid != 0:  # parent

    # child is in its own process group, manually pass kill signals
    signal.signal(signal.SIGINT, lambda signum, frame: os.kill(child_pid, signal.SIGINT))
    signal.signal(signal.SIGTERM, lambda signum, frame: os.kill(child_pid, signal.SIGTERM))

    fcntl.fcntl(sys.stdout, fcntl.F_SETFL, fcntl.fcntl(sys.stdout, fcntl.F_GETFL) | os.O_NONBLOCK)

    while True:
      try:
        dat = os.read(child_pty, 4096)
      except OSError as e:
        if e.errno == errno.EIO:
          break
        continue

      if not dat:
        break

      try:
        sys.stdout.write(dat.decode('utf8'))
      except (OSError, IOError, UnicodeDecodeError):
        pass

    # os.wait() returns a tuple with the pid and a 16 bit value
    # whose low byte is the signal number and whose high byte is the exit satus
    exit_status = os.wait()[1] >> 8
    os._exit(exit_status)


def heartbeat():
  work_dir = '/data/openpilot'

  while True:
    try:
      with open(os.path.join(work_dir, "selfdrive", "common", "version.h")) as _versionf:
        version = _versionf.read().split('"')[1]

      tmux = ""

      # try:
      #  tmux = os.popen('tail -n 100 /tmp/tmux_out').read()
      # except Exception:
      #  pass

      params = Params()
      msg = {
        'version': version,
        'dongle_id': params.get("DongleId").rstrip().decode('utf8'),
        'remote': subprocess.check_output(["git", "config", "--get", "remote.origin.url"], cwd=work_dir).decode('utf8').rstrip(),
        'branch': subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=work_dir).decode('utf8').rstrip(),
        'revision': subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=work_dir).decode('utf8').rstrip(),
        'serial': subprocess.check_output(["getprop", "ro.boot.serialno"]).decode('utf8').rstrip(),
        'tmux': tmux,
      }
      with Timeout(10):
        requests.post('http://%s/eon/heartbeat/' % HOST, json=msg, timeout=5.0)
    except Exception as e:
      print("Unable to send heartbeat", e)

    time.sleep(5)


if __name__ == "__main__":
  unblock_stdout()
  heartbeat()
