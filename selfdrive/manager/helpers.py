import os
import sys
import fcntl
import errno
import signal
import shutil
import subprocess
import tempfile
import threading

from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params

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


def write_onroad_params(started, params):
  params.put_bool("IsOnroad", started)
  params.put_bool("IsOffroad", not started)


def save_bootlog():
  # copy current params
  tmp = tempfile.mkdtemp()
  shutil.copytree(Params().get_param_path() + "/..", tmp, dirs_exist_ok=True)

  def fn(tmpdir):
    env = os.environ.copy()
    env['PARAMS_ROOT'] = tmpdir
    subprocess.call("./bootlog", cwd=os.path.join(BASEDIR, "system/loggerd"), env=env)
    shutil.rmtree(tmpdir)
  t = threading.Thread(target=fn, args=(tmp, ))
  t.daemon = True
  t.start()
