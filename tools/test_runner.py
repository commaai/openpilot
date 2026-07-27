#!/usr/bin/env python3
"""Fast, friendly, parallel unittest runner."""

import argparse
from collections import Counter, OrderedDict
from concurrent.futures import as_completed, ProcessPoolExecutor
import contextlib
import math
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
import unittest
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
IGNORED = (
  ROOT / "openpilot/selfdrive/test/process_replay/test_processes.py",
  ROOT / "openpilot/selfdrive/test/process_replay/test_regen.py",
  ROOT / "openpilot/tools/sim",
)
BAD = {"failed", "error", "xpassed"}
MARK = {
  "passed": (".", 32), "skipped": ("s", 33), "xfailed": ("x", 36),
  "failed": ("F", 31), "error": ("E", 31), "xpassed": ("X", 31),
}
CAPTURE_OUTPUT = True
COLOR = False

def paint(text, code):
  return f"\033[{code}m{text}\033[0m" if COLOR else text
def cpu_count():
  with contextlib.suppress(AttributeError, OSError):
    return len(os.sched_getaffinity(0))
  return os.cpu_count() or 1
def parse_jobs(value):
  if value in ("auto", "logical"):
    return cpu_count()
  try:
    value = int(value)
    if value < 1:
      raise ValueError
    return value
  except (AssertionError, ValueError) as e:
    raise argparse.ArgumentTypeError("expected a positive integer or 'auto'") from e

class Capture:
  """Capture Python, native, and subprocess writes to stdout/stderr."""
  def start(self):
    if not CAPTURE_OUTPUT:
      return
    for stream in (sys.stdout, sys.stderr):
      with contextlib.suppress(Exception):
        stream.flush()
    self.files = [tempfile.TemporaryFile() for _ in range(2)]
    self.saved = [os.dup(fd) for fd in (1, 2)]
    for fd, file in zip((1, 2), self.files, strict=True):
      os.dup2(file.fileno(), fd)
  def stop(self, keep=True):
    if not CAPTURE_OUTPUT:
      return "", ""
    for stream in (sys.stdout, sys.stderr):
      with contextlib.suppress(Exception):
        stream.flush()
    for fd, saved in zip((1, 2), self.saved, strict=True):
      os.dup2(saved, fd)
      os.close(saved)
    output = []
    for file in self.files:
      if keep:
        file.seek(0)
        output.append(file.read().decode(errors="replace"))
      file.close()
    return output if keep else ("", "")
def make_record(test_id, status="passed", detail=""):
  return {"id": test_id, "status": status, "detail": detail, "time": 0., "stdout": "", "stderr": ""}
class Result(unittest.TestResult):
  def __init__(self):
    super().__init__()
    self.records, self.current = [], None
  def startTest(self, test):
    self.current, self.started, self.capture = make_record(test.id()), time.monotonic(), Capture()
    self.capture.start()
  def stopTest(self, test):
    assert self.current is not None
    self.current["time"] = time.monotonic() - self.started
    keep = self.current["status"] in BAD
    stdout, stderr = self.capture.stop(keep)
    if keep:
      self.current["stdout"], self.current["stderr"] = stdout, stderr
    self.records.append(self.current)
    self.current = None
  def mark(self, test, status, detail=""):
    if self.current is None:  # setUpClass/setUpModule can fail before a test starts
      self.records.append(make_record(test.id(), status, detail))
      return
    priority = {"passed": 0, "skipped": 1, "xfailed": 1, "failed": 2, "xpassed": 2, "error": 3}
    if priority[status] >= priority[self.current["status"]]:
      self.current["status"] = status
    if detail:
      self.current["detail"] += ("\n\n" if self.current["detail"] else "") + detail
  def addFailure(self, test, err):
    self.mark(test, "failed", self._exc_info_to_string(err, test))
  def addError(self, test, err):
    self.mark(test, "error", self._exc_info_to_string(err, test))
  def addSkip(self, test, reason):
    self.mark(test, "skipped", reason)
  def addExpectedFailure(self, test, err):
    self.mark(test, "xfailed", self._exc_info_to_string(err, test))
  def addUnexpectedSuccess(self, test):
    self.mark(test, "xpassed", "Test was expected to fail, but passed.")
  def addSubTest(self, test, subtest, err):
    if err:
      status = "failed" if issubclass(err[0], test.failureException) else "error"
      self.mark(test, status, f"{subtest}\n{self._exc_info_to_string(err, test)}")
def flatten(suite):
  for test in suite:
    yield from flatten(test) if isinstance(test, unittest.TestSuite) else (test,)
def module_name(path):
  relative = path.resolve().relative_to(ROOT).with_suffix("")
  if not all(part.isidentifier() for part in relative.parts):
    raise ValueError(f"{path}: path cannot be imported as a Python module")
  return ".".join(relative.parts)
def collect(targets, keyword, use_ignores):
  loader, tests, errors, names = unittest.TestLoader(), [], [], []
  for target in targets:
    path_text, *nodes = target.split("::")
    path = Path(path_text)
    try:
      if path.is_dir():
        files = sorted(path.rglob("test_*.py"))
        if use_ignores:
          files = [f for f in files if not any(f.resolve().is_relative_to(i) for i in IGNORED)]
        names.extend(module_name(file) for file in files)
      elif path.is_file():
        names.append(".".join((module_name(path), *nodes)))
      elif "/" in path_text or path_text.endswith(".py"):
        errors.append(f"{target}: file or directory not found")
      else:
        names.append(target.replace("::", "."))
    except (OSError, ValueError) as e:
      errors.append(str(e))
  for name in dict.fromkeys(names):
    before = len(loader.errors)
    try:
      suite = loader.loadTestsFromName(name)
    except Exception:
      errors.append(f"Failed to collect {name}\n{traceback.format_exc()}")
      continue
    errors.extend(loader.errors[before:])
    for test in flatten(suite):
      cls = type(test)
      if cls.__name__ == "_FailedTest":
        continue
      if getattr(cls, "__unittest_skip_why__", "") == "parameterized base class":
        continue
      if os.environ.get("SKIP_SLOW") and getattr(cls, "SLOW_TEST", False):
        continue
      if not keyword or keyword.lower() in test.id().lower():
        tests.append(test)
  return list({test.id(): test for test in tests}.values()), errors
def make_batches(tests, workers):
  groups, atomic = OrderedDict(), set()
  for test in tests:
    cls, module = type(test), sys.modules.get(type(test).__module__)
    key = f"{cls.__module__}.{cls.__qualname__}"
    if module and (hasattr(module, "setUpModule") or hasattr(module, "tearDownModule")):
      key = cls.__module__
      atomic.add(key)
    if "setUpClass" in cls.__dict__ or "tearDownClass" in cls.__dict__:
      atomic.add(key)
    groups.setdefault(key, []).append(test.id())
  size, batches, pending = max(1, math.ceil(len(tests) / max(1, workers * 4))), [], []
  for key, ids in groups.items():
    if key in atomic:
      if pending:
        batches.append(pending)
        pending = []
      batches.append(ids)
      continue
    while ids:
      take = size - len(pending)
      pending.extend(ids[:take])
      ids = ids[take:]
      if len(pending) == size:
        batches.append(pending)
        pending = []
  return sorted(batches + ([pending] if pending else []), key=len, reverse=True)
def run_batch(test_ids):
  result, outside = Result(), Capture()
  try:
    os.chdir(ROOT)
    outside.start()
    try:
      unittest.TestLoader().loadTestsFromNames(test_ids).run(result)
    finally:
      stdout, stderr = outside.stop()
    failures = [item for item in result.records if item["status"] in BAD]
    if failures:  # attach class/module fixture output to the first related failure
      failures[0]["stdout"] = stdout + failures[0]["stdout"]
      failures[0]["stderr"] = stderr + failures[0]["stderr"]
    return result.records
  except Exception:
    with contextlib.suppress(Exception):
      outside.stop()
    return [make_record(test_ids[0], "error", traceback.format_exc())]

def init_worker(warning_action, capture_output):
  global CAPTURE_OUTPUT
  CAPTURE_OUTPUT = capture_output
  warnings.simplefilter(warning_action)
def run_parallel(batches, workers, warning_action, capture_output):
  context = multiprocessing.get_context()
  with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=init_worker,
                           initargs=(warning_action, capture_output)) as pool:
    futures = {pool.submit(run_batch, batch): batch for batch in batches}
    for future in as_completed(futures):
      try:
        yield future.result()
      except Exception:
        yield [make_record(futures[future][0], "error", traceback.format_exc())]
def show_result(item, verbose, column):
  mark, code = MARK[item["status"]]
  if verbose:
    print(f"{paint(mark, code)} {item['id']} {item['time']:.2f}s")
    return 0
  print(paint(mark, code), end="", flush=True)
  if column == 79:
    print()
    return 0
  return column + 1
def report(records, errors, duration_count, elapsed):
  width = min(100, os.get_terminal_size().columns if sys.stdout.isatty() else 80)
  for index, error in enumerate(errors, 1):
    print(paint(f"\n{'=' * 8} COLLECTION ERROR {index} {'=' * 8}", 31))
    print(error.rstrip())
  for item in sorted((r for r in records if r["status"] in BAD), key=lambda r: r["id"]):
    print(paint(f"\n{f' {item['status'].upper()} {item['id']} ':=^{width}}", 31))
    if item["detail"]:
      print(item["detail"].rstrip())
    for stream in ("stdout", "stderr"):
      if item[stream]:
        print(paint(f"\n--- captured {stream} ---", 33))
        print(item[stream].rstrip())
  timed = sorted((r for r in records if r["time"]), key=lambda r: r["time"], reverse=True)
  timed = timed if duration_count == 0 else timed[:duration_count]
  if timed:
    print(paint("\nslowest tests", 36))
    for item in timed:
      print(f"{item['time']:8.2f}s  {item['id']}")
  counts = Counter(item["status"] for item in records)
  parts = [f"{counts[name]} {name}" for name in MARK if counts[name]]
  if errors:
    parts.append(f"{len(errors)} collection error{'s' if len(errors) != 1 else ''}")
  failed = bool(errors) or any(counts[name] for name in BAD)
  print(paint(f"\n{', '.join(parts) or 'no tests ran'} in {elapsed:.2f}s", 31 if failed else 32))
  return 1 if failed else (0 if records else 5)

def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("targets", nargs="*", help="files, directories, dotted IDs, or path.py::Class::test")
  parser.add_argument("-j", "--jobs", type=parse_jobs, default=cpu_count(), help="workers (default: available CPUs)")
  parser.add_argument("-k", metavar="TEXT", help="only run test IDs containing TEXT")
  parser.add_argument("-s", "--no-capture", action="store_true", help="show test output live")
  parser.add_argument("-v", "--verbose", action="store_true", help="show every test")
  parser.add_argument("--durations", type=int, default=10, metavar="N", help="show N slowest tests; 0 shows all")
  parser.add_argument("-W", "--warnings", choices=("error", "default", "always", "ignore"), default="error")
  args = parser.parse_args(argv)
  if args.durations < 0:
    parser.error("--durations must be non-negative")
  global CAPTURE_OUTPUT, COLOR
  CAPTURE_OUTPUT, COLOR = not args.no_capture, sys.stdout.isatty() and "NO_COLOR" not in os.environ
  os.chdir(ROOT)
  warnings.simplefilter(args.warnings)
  started, implicit = time.monotonic(), not args.targets
  tests, errors = collect(args.targets or ["openpilot"], args.k, implicit)
  batches = make_batches(tests, args.jobs)
  workers = min(args.jobs, len(batches))
  print(f"collected {len(tests)} test{'s' if len(tests) != 1 else ''} in {time.monotonic() - started:.2f}s" +
        f" • {workers} worker{'s' if workers != 1 else ''}")
  records, column = [], 0
  try:
    streams = map(run_batch, batches) if workers < 2 else run_parallel(batches, workers, args.warnings, CAPTURE_OUTPUT)
    for batch in streams:
      records.extend(batch)
      for item in batch:
        column = show_result(item, args.verbose, column)
  except KeyboardInterrupt:
    print(paint("\ninterrupted", 31))
    return 2
  if column:
    print()
  return report(records, errors, args.durations, time.monotonic() - started)

if __name__ == "__main__":
  raise SystemExit(main())
