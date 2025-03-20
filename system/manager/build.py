#!/usr/bin/env python3
import os
import subprocess
import select
from pathlib import Path
import pyray as rl

# NOTE: Do NOT import anything here that needs be built (e.g. params)
from openpilot.common.basedir import BASEDIR
from openpilot.system.hardware import HARDWARE, AGNOS
from openpilot.common.swaglog import cloudlog, add_file_handler
from openpilot.system.version import get_build_metadata
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.spinner import Spinner
from openpilot.system.ui.text import TextWindow

MAX_CACHE_SIZE = 4e9 if "CI" in os.environ else 2e9
CACHE_DIR = Path("/data/scons_cache" if AGNOS else "/tmp/scons_cache")

TOTAL_SCONS_NODES = 3130
MAX_BUILD_PROGRESS = 100

def build(dirty: bool = False, minimal: bool = False) -> None:
  env = os.environ.copy()
  env['SCONS_PROGRESS'] = "1"
  nproc = os.cpu_count() or 2

  extra_args = ["--minimal"] if minimal else []
  if AGNOS:
    HARDWARE.set_power_save(False)
    os.sched_setaffinity(0, range(8))  # ensure we can use the isolcpus cores

  # building with all cores can result in using too
  # much memory, so retry with less parallelism
  gui_app.init_window("Spinner")
  spinner = Spinner()
  compile_output: list[bytes] = []
  for n in (nproc, nproc/2, 1):
    compile_output.clear()
    scons: subprocess.Popen = subprocess.Popen(["scons", f"-j{int(n)}", "--cache-populate", *extra_args], cwd=BASEDIR, env=env, stderr=subprocess.PIPE)
    assert scons.stderr is not None
    os.set_blocking(scons.stderr.fileno(), False)  # Non-blocking reads

    # Read progress from stderr and update spinner
    spinner.set_text("0")
    while scons.poll() is None:
      try:
        rl.begin_drawing()
        rl.clear_background(rl.BLACK)
        spinner.render()
        rl.end_drawing()

        if scons.stderr in select.select([scons.stderr], [], [], 0.02)[0]:
          line = scons.stderr.readline()
          if not line:
            continue
        line = line.rstrip()
        prefix = b'progress: '
        if line.startswith(prefix):
          i = int(line[len(prefix):])
          spinner.set_text(str(int(MAX_BUILD_PROGRESS * min(1., i / TOTAL_SCONS_NODES))))
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
      compile_output += [line for line in scons.stderr.read().split(b'\n') if not line.startswith(b'progress')]

    # Build failed log errors
    error_s = b"\n".join(compile_output).decode('utf8', 'replace')
    add_file_handler(cloudlog)
    cloudlog.error("scons build failed\n" + error_s)

    # Show TextWindow
    if not os.getenv("CI"):
      text_window = TextWindow("openpilot failed to build\n \n" + error_s)
      while True:
        rl.begin_drawing()
        rl.clear_background(rl.BLACK)
        text_window.render()
        rl.end_drawing()
    exit(1)

  # enforce max cache size
  cache_files = [f for f in CACHE_DIR.rglob('*') if f.is_file()]
  cache_files.sort(key=lambda f: f.stat().st_mtime)
  cache_size = sum(f.stat().st_size for f in cache_files)
  for f in cache_files:
    if cache_size < MAX_CACHE_SIZE:
      break
    cache_size -= f.stat().st_size
    f.unlink()


if __name__ == "__main__":
  build_metadata = get_build_metadata()
  build(build_metadata.openpilot.is_dirty, minimal = AGNOS)
