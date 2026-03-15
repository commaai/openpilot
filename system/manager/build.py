#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

# NOTE: Do NOT import anything here that needs be built (e.g. params)
from openpilot.common.basedir import BASEDIR
from openpilot.common.spinner import Spinner
from openpilot.common.text_window import TextWindow
from openpilot.common.swaglog import cloudlog, add_file_handler
from openpilot.system.hardware import HARDWARE, AGNOS
from openpilot.system.version import get_build_metadata

MAX_CACHE_SIZE = 4e9 if "CI" in os.environ else 2e9
CACHE_DIR = Path("/data/scons_cache" if AGNOS else "/tmp/scons_cache")

TOTAL_SCONS_NODES = 2705
MAX_BUILD_PROGRESS = 100

def get_scons_cache_lock_path() -> Path:
  return CACHE_DIR / "config.lock"

def clear_stale_scons_cache_lock(build_output: list[bytes]) -> bool:
  lock_path = get_scons_cache_lock_path()
  lock_path_b = str(lock_path).encode()
  if not any(b"SConsLockFailure" in line and lock_path_b in line for line in build_output):
    return False

  try:
    lock_path.unlink()
  except FileNotFoundError:
    return False
  except OSError:
    cloudlog.exception(f"failed to remove stale scons cache lock {lock_path}")
    return False

  cloudlog.warning(f"removed stale scons cache lock {lock_path}, retrying build")
  return True

def build(spinner: Spinner, dirty: bool = False, minimal: bool = False) -> None:
  env = os.environ.copy()
  env['SCONS_PROGRESS'] = "1"
  nproc = os.cpu_count()
  if nproc is None:
    nproc = 2

  extra_args = ["--minimal"] if minimal else []

  if AGNOS:
    HARDWARE.set_power_save(False)
    os.sched_setaffinity(0, range(8))  # ensure we can use the isolcpus cores

  # building with all cores can result in using too
  # much memory, so retry with less parallelism
  compile_output: list[bytes] = []
  retry_job_counts = [nproc, nproc/2, 1]
  retried_after_lock_cleanup = False
  i = 0
  while i < len(retry_job_counts):
    n = retry_job_counts[i]
    compile_output.clear()
    scons: subprocess.Popen = subprocess.Popen(["scons", f"-j{int(n)}", "--cache-populate", *extra_args], cwd=BASEDIR, env=env, stderr=subprocess.PIPE)
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
          i = int(line[len(prefix):])
          spinner.update_progress(MAX_BUILD_PROGRESS * min(1., i / TOTAL_SCONS_NODES), 100.)
        elif len(line):
          compile_output.append(line)
          print(line.decode('utf8', 'replace'))
      except Exception:
        pass

    compile_output += [line for line in scons.stderr.read().split(b'\n') if len(line)]

    if scons.returncode == 0:
      break

    if not retried_after_lock_cleanup and clear_stale_scons_cache_lock(compile_output):
      retried_after_lock_cleanup = True
      continue

    i += 1

  if scons.returncode != 0:
    # Build failed log errors
    error_s = b"\n".join(compile_output).decode('utf8', 'replace')
    add_file_handler(cloudlog)
    cloudlog.error("scons build failed\n" + error_s)

    # Show TextWindow
    spinner.close()
    if not os.getenv("CI"):
      with TextWindow("openpilot failed to build\n \n" + error_s) as t:
        t.wait_for_exit()
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
  spinner = Spinner()
  spinner.update_progress(0, 100)
  build_metadata = get_build_metadata()
  build(spinner, build_metadata.openpilot.is_dirty, minimal = AGNOS)
