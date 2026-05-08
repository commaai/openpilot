#!/usr/bin/env python3
import os
import subprocess

# NOTE: Do NOT import anything here that needs be built (e.g. params)
from openpilot.common.basedir import BASEDIR
from openpilot.common.spinner import Spinner
from openpilot.common.text_window import TextWindow
from openpilot.system.hardware import HARDWARE, AGNOS

def build() -> None:
  spinner = Spinner()
  spinner.update_progress(0, 100)
  env = os.environ.copy()

  HARDWARE.set_power_save(False)
  if AGNOS:
    os.sched_setaffinity(0, range(8))  # ensure we can use the isolcpus cores

  # building with all cores can result in using too much memory, so retry serially
  compile_output: list[bytes] = []
  for parallelism in ([], ["-j4"], ["-j1"]):
    compile_output.clear()
    scons: subprocess.Popen = subprocess.Popen(["scons", *parallelism], cwd=BASEDIR, env=env, stderr=subprocess.PIPE)
    assert scons.stderr is not None

    # Read progress from stderr and update spinner
    while scons.poll() is None:
      try:
        line = scons.stderr.readline()
        if line is None:
          continue
        line = line.rstrip()

        prefix = b'progress: '
        if line.startswith(prefix):
          progress = float(line[len(prefix):])
          spinner.update_progress(100 * min(1., progress / 100.), 100.)
        elif len(line):
          compile_output.append(line)
          print(line.decode('utf8', 'replace'))
      except Exception:
        pass

    if scons.returncode == 0:
      break

  if scons.returncode != 0:
    # Read remaining output
    if scons.stderr is not None:
      compile_output += scons.stderr.read().split(b'\n')

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
