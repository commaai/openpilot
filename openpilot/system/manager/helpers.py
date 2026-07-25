import errno
import fcntl
import os
import sys
import shutil
import signal
import tempfile
import threading

from openpilot.common.params import Params
from openpilot.system.loggerd.bootlog import create_bootlog

def unblock_stdout() -> None:
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
      except (OSError, UnicodeDecodeError):
        pass

    # os.wait() returns a tuple with the pid and a 16 bit value
    # whose low byte is the signal number and whose high byte is the exit status
    exit_status = os.wait()[1] >> 8
    os._exit(exit_status)

def save_bootlog():
  # copy current params
  tmp = tempfile.mkdtemp()
  params_path = Params().get_param_path()
  shutil.copytree(params_path, os.path.join(tmp, os.path.basename(params_path)), dirs_exist_ok=True)

  def run_bootlog(tmpdir):
    try:
      create_bootlog(tmpdir)
    finally:
      shutil.rmtree(tmpdir)

  threading.Thread(target=run_bootlog, args=(tmp, ), daemon=True).start()
