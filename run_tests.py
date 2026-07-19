#!/usr/bin/env python3
# openpilot's test runner: stdlib unittest, one process per test file, live prefixed output.
import argparse
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

BASEDIR = Path(__file__).resolve().parent

# heavy CI-only tests, invoked explicitly in .github/workflows/tests.yaml
EXCLUDES = (
  "openpilot/selfdrive/test/process_replay/test_processes.py",
  "openpilot/selfdrive/test/process_replay/test_regen.py",
  "openpilot/tools/sim/",
)

# wraps C++ test binaries in an OpenpilotPrefix
CPP_HARNESS = BASEDIR / "openpilot/selfdrive/test/cpp_harness.py"

COLORS = [32, 33, 34, 35, 36, 92, 93, 94, 95, 96]  # ANSI
RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
BOLD = "\033[1m"

RESULT_RE = re.compile(r"^Ran (\d+) tests? in")


def use_color() -> bool:
  return "NO_COLOR" not in os.environ and (sys.stdout.isatty() or "FORCE_COLOR" in os.environ)


def is_elf_executable(p: Path) -> bool:
  if not (p.is_file() and os.access(p, os.X_OK)):
    return False
  with open(p, 'rb') as f:
    return f.read(4) == b"\x7fELF"


def discover(paths: list[str]) -> list[Path]:
  tests = []
  if paths:
    for p in (Path(p) for p in paths):
      if p.is_dir():
        tests.extend(sorted(p.rglob("test_*.py")))
        tests.extend(sorted(f for f in p.rglob("test_*") if f.suffix == "" and is_elf_executable(f)))
      else:
        tests.append(p)
  else:
    for f in sorted((BASEDIR / "openpilot").rglob("test_*")):
      rel = f.relative_to(BASEDIR).as_posix()
      if any(rel == e or rel.startswith(e) for e in EXCLUDES):
        continue
      if f.suffix == ".py" or (f.suffix == "" and is_elf_executable(f)):
        tests.append(f)
  return [t.resolve() for t in tests]


class Job:
  def __init__(self, path: Path, idx: int):
    self.path = path
    self.name = path.relative_to(BASEDIR).as_posix()
    self.color = COLORS[idx % len(COLORS)]
    self.proc: subprocess.Popen | None = None
    self.ret: int | None = None
    self.duration = 0.0
    self.n_ran = 0
    self.timed_out = False

  @property
  def is_cpp(self) -> bool:
    return self.path.suffix != ".py"

  def cmd(self, k_patterns: list[str]) -> list[str]:
    if self.is_cpp:
      return [sys.executable, str(CPP_HARNESS), str(self.path)]
    module = self.path.relative_to(BASEDIR).with_suffix("").as_posix().replace("/", ".")
    cmd = [sys.executable, "-u", "-W", "error", "-m", "unittest", "-v", "--durations", "10", module]
    for k in k_patterns:
      cmd += ["-k", k]
    return cmd

  @property
  def failed(self) -> bool:
    # unittest exits 5 when no tests matched -k; that's not a failure
    return self.ret not in (0, 5) or self.timed_out


class Runner:
  def __init__(self, args):
    self.args = args
    self.print_lock = threading.Lock()
    self.procs_lock = threading.Lock()
    self.interrupted = False
    self.color = use_color()

  def _print(self, job: Job, line: str):
    prefix = f"[{job.name}]"
    if self.color:
      prefix = f"\033[{job.color}m{prefix}{RESET}"
    with self.print_lock:
      print(f"{prefix} {line}", flush=True)

  def run_job(self, job: Job) -> None:
    if self.interrupted:
      return
    env = os.environ.copy()
    if self.args.explicit:
      env["RUN_SLOW"] = "1"

    start = time.monotonic()
    job.proc = subprocess.Popen(job.cmd(self.args.k or []), cwd=BASEDIR, env=env, text=True, errors="replace",
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)

    def _kill():
      job.timed_out = True
      with self.procs_lock:
        if job.proc.poll() is None:
          os.killpg(job.proc.pid, signal.SIGKILL)
    watchdog = threading.Timer(self.args.timeout, _kill) if self.args.timeout else None
    if watchdog is not None:
      watchdog.start()
    try:
      assert job.proc.stdout is not None
      for line in job.proc.stdout:
        line = line.rstrip("\n")
        if (m := RESULT_RE.match(line)) is not None:
          job.n_ran = int(m.group(1))
        self._print(job, line)
      job.ret = job.proc.wait()
    finally:
      if watchdog is not None:
        watchdog.cancel()
    job.duration = time.monotonic() - start

    if job.timed_out:
      self._print(job, f"{RED if self.color else ''}TIMED OUT after {self.args.timeout}s{RESET if self.color else ''}")

  def run(self, jobs: list[Job]) -> int:
    start = time.monotonic()
    threads: list[threading.Thread] = []
    pending = list(jobs)
    sem = threading.Semaphore(self.args.jobs)

    def worker(job: Job):
      with sem:
        self.run_job(job)

    try:
      for job in pending:
        t = threading.Thread(target=worker, args=(job,), daemon=True)
        t.start()
        threads.append(t)
      for t in threads:
        while t.is_alive():
          t.join(0.1)
    except KeyboardInterrupt:
      self.interrupted = True
      with self.procs_lock:
        for job in jobs:
          if job.proc is not None and job.proc.poll() is None:
            os.killpg(job.proc.pid, signal.SIGKILL)
      print("\ninterrupted")
      return 130

    return self.summarize(jobs, time.monotonic() - start)

  def summarize(self, jobs: list[Job], total_time: float) -> int:
    done = [j for j in jobs if j.ret is not None or j.timed_out]
    failed = [j for j in done if j.failed]
    n_tests = sum(j.n_ran for j in done)

    print(f"\n{BOLD if self.color else ''}{'='*40} summary {'='*40}{RESET if self.color else ''}")
    for j in sorted(done, key=lambda j: -j.duration)[:15]:
      print(f"{j.duration:8.2f}s  {j.name}")
    for j in failed:
      print(f"{RED if self.color else ''}FAILED{RESET if self.color else ''}  {j.name}" + (" (timed out)" if j.timed_out else f" (exit {j.ret})"))

    ok = not failed and len(done) == len(jobs)
    status = f"{GREEN}PASSED{RESET}" if ok else f"{RED}FAILED{RESET}"
    if not self.color:
      status = "PASSED" if ok else "FAILED"
    print(f"{status}: {len(done) - len(failed)}/{len(jobs)} files, {n_tests} tests, in {total_time:.2f}s")
    return 0 if ok else 1


def list_tests(jobs: list[Job]) -> int:
  # import every test module in one process: reports collection errors and precompiles bytecode
  modules = [j.cmd([])[-1] for j in jobs if not j.is_cpp]
  script = """
import importlib, sys
failed = False
for m in sys.stdin.read().split():
  try:
    importlib.import_module(m)
    print(m)
  except Exception as e:
    failed = True
    print(f'IMPORT FAILED: {m}: {e!r}', file=sys.stderr)
sys.exit(failed)
"""
  ret = subprocess.run([sys.executable, "-c", script], cwd=BASEDIR, input="\n".join(modules), text=True).returncode
  for j in jobs:
    if j.is_cpp:
      print(j.name)
  return ret


def main() -> int:
  parser = argparse.ArgumentParser(description="openpilot's parallel unittest runner")
  parser.add_argument("paths", nargs="*", help="test files or directories (default: all of openpilot/)")
  parser.add_argument("-k", action="append", help="only run tests matching this substring/fnmatch pattern (repeatable, ORed; matches class and method names)")
  parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count(), help="max parallel test files")
  parser.add_argument("--timeout", type=float, default=None, help="per-file timeout in seconds (default: 600 for full runs, none for explicit files)")
  parser.add_argument("--list", action="store_true", help="import all test modules and list them without running")
  args = parser.parse_args()
  args.explicit = any(Path(p).is_file() for p in args.paths)
  if args.timeout is None and not args.explicit:
    args.timeout = 600.

  tests = discover(args.paths)
  if not tests:
    print("no tests found")
    return 1
  jobs = [Job(p, i) for i, p in enumerate(tests)]

  if args.list:
    return list_tests(jobs)
  return Runner(args).run(jobs)


if __name__ == "__main__":
  sys.exit(main())
