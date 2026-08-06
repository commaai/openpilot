#!/usr/bin/env python3
import os
import subprocess
import time

# NOTE: Do NOT import anything here that needs be built (e.g. params)
from openpilot.common.basedir import BASEDIR
from openpilot.common.spinner import Spinner
from openpilot.common.text_window import TextWindow
from openpilot.common.hardware import HARDWARE, AGNOS

PHASE_TIMEOUT = 30.  # keep a long action's message up this long after its last output

def build() -> None:
  spinner = Spinner()
  spinner.update_progress(0, 100)

  HARDWARE.set_power_save(False)
  if AGNOS:
    os.sched_setaffinity(0, range(8))  # ensure we can use the isolcpus cores

  # building with all cores can result in using too much memory, so retry serially
  compile_output: list[bytes] = []
  for parallelism in ([], ["-j4"], ["-j1"]):
    compile_output.clear()
    with subprocess.Popen(["scons", *parallelism], cwd=BASEDIR, env={**os.environ, "PWD": BASEDIR}, stderr=subprocess.PIPE) as scons:
      assert scons.stderr is not None

      # Read progress from stderr and update spinner
      phase_deadline = 0.
      while scons.poll() is None:
        try:
          line = scons.stderr.readline()
          if line is None:
            continue
          line = line.rstrip()

          prefix = b'progress: '
          text_prefix = b'spinner: '
          now = time.monotonic()
          if line.startswith(text_prefix):
            # a long action took over the spinner, keep its message up while it's still logging
            spinner.update(line[len(text_prefix):].decode('utf8', 'replace'))
            phase_deadline = now + PHASE_TIMEOUT
          elif line.startswith(prefix):
            if now > phase_deadline:
              progress = float(line[len(prefix):])
              spinner.update_progress(100 * min(1., progress / 100.), 100.)
          elif len(line):
            if now < phase_deadline:
              phase_deadline = now + PHASE_TIMEOUT
            compile_output.append(line)
            print(line.decode('utf8', 'replace'))
        except Exception:
          pass

      # Drain and close the pipe before retrying or returning.
      for line in scons.stderr.read().split(b'\n'):
        line = line.rstrip()
        if len(line):
          compile_output.append(line)

    if scons.returncode == 0:
      break

  if scons.returncode != 0:
    # Build failed log errors
    error_s = b"\n".join(compile_output).decode('utf8', 'replace')

    # Show TextWindow
    spinner.close()
    if not os.getenv("CI"):
      with TextWindow("openpilot failed to build\n \n" + error_s) as t:
        t.wait_for_exit()
    exit(1)

if __name__ == "__main__":
  build()
