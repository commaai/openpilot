import os
import sys
import shutil
import subprocess
import textwrap
import time
import fcntl
import errno
import signal

from selfdrive.swaglog import cloudlog, add_logentries_handler
from common.text_window import TextWindow
from common.basedir import BASEDIR

TOTAL_SCONS_NODES = 1225
MAX_BUILD_PROGRESS = 70

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


def build(spinner, dirty=False):
  env = os.environ.copy()
  env['SCONS_PROGRESS'] = "1"
  env['SCONS_CACHE'] = "1"
  nproc = os.cpu_count()
  j_flag = "" if nproc is None else f"-j{nproc - 1}"

  for retry in [True, False]:
    scons = subprocess.Popen(["scons", j_flag], cwd=BASEDIR, env=env, stderr=subprocess.PIPE)

    compile_output = []

    # Read progress from stderr and update spinner
    while scons.poll() is None:
      try:
        line = scons.stderr.readline()
        if line is None:
          continue
        line = line.rstrip()

        prefix = b'progress: '
        if line.startswith(prefix):
          i = int(line[len(prefix):])
          spinner.update_progress(MAX_BUILD_PROGRESS * min(1., i / TOTAL_SCONS_NODES), 100.)
        elif len(line):
          compile_output.append(line)
          print(line.decode('utf8', 'replace'))
      except Exception:
        pass

    if scons.returncode != 0:
      # Read remaining output
      r = scons.stderr.read().split(b'\n')
      compile_output += r

      if retry and (not dirty):
        if not os.getenv("CI"):
          print("scons build failed, cleaning in")
          for i in range(3, -1, -1):
            print("....%d" % i)
            time.sleep(1)
          subprocess.check_call(["scons", "-c"], cwd=BASEDIR, env=env)
          shutil.rmtree("/tmp/scons_cache", ignore_errors=True)
          shutil.rmtree("/data/scons_cache", ignore_errors=True)
        else:
          print("scons build failed after retry")
          sys.exit(1)
      else:
        # Build failed log errors
        errors = [line.decode('utf8', 'replace') for line in compile_output
                  if any([err in line for err in [b'error: ', b'not found, needed by target']])]
        error_s = "\n".join(errors)
        add_logentries_handler(cloudlog)
        cloudlog.error("scons build failed\n" + error_s)

        # Show TextWindow
        spinner.close()
        error_s = "\n \n".join(["\n".join(textwrap.wrap(e, 65)) for e in errors])
        with TextWindow("openpilot failed to build\n \n" + error_s) as t:
          t.wait_for_exit()
        exit(1)
    else:
      break
