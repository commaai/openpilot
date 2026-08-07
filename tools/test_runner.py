#!/usr/bin/env python3
import argparse
from collections import Counter
from concurrent.futures import as_completed, ProcessPoolExecutor
from itertools import batched
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
import unittest
import warnings

ROOT = Path(__file__).resolve().parents[1]
IGNORED = (
  ROOT / "openpilot/selfdrive/test/process_replay/test_processes.py",
  ROOT / "openpilot/tools/sim",
)
FAILURES = {"failed", "error", "xpassed"}
STATUS_MARKS = {
  "passed": (".", 32),
  "skipped": ("s", 33),
  "xfailed": ("x", 36),
  "failed": ("F", 31),
  "error": ("E", 31),
  "xpassed": ("X", 31),
}


def paint(text, code):
  if sys.stdout.isatty() and "NO_COLOR" not in os.environ:
    return f"\033[{code}m{text}\033[0m"
  return text


class Capture:
  def __init__(self, enabled):
    self.enabled = enabled

  def start(self):
    if not self.enabled:
      return
    for stream in (sys.stdout, sys.stderr):
      stream.flush()
    self.files = [tempfile.TemporaryFile() for _ in range(2)]
    self.saved = [os.dup(fd) for fd in (1, 2)]
    for fd, file in enumerate(self.files, 1):
      os.dup2(file.fileno(), fd)

  def stop(self, keep=True):
    if not self.enabled:
      return "", ""
    for stream in (sys.stdout, sys.stderr):
      stream.flush()
    for fd, saved in enumerate(self.saved, 1):
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
  return {"id": test_id, "status": status, "detail": detail, "time": 0.0, "stdout": "", "stderr": ""}


class Result(unittest.TestResult):
  def __init__(self, capture_output):
    super().__init__()
    self.records = []
    self.current = None
    self.capture = Capture(capture_output)

  def startTest(self, test):
    self.current = make_record(test.id())
    self.started = time.monotonic()
    self.capture.start()

  def stopTest(self, test):
    self.current["time"] = time.monotonic() - self.started
    keep = self.current["status"] in FAILURES
    stdout, stderr = self.capture.stop(keep)
    if keep:
      self.current["stdout"] = stdout
      self.current["stderr"] = stderr
    self.records.append(self.current)
    self.current = None

  def mark(self, test, status, detail=""):
    if self.current is None:  # setUpClass/setUpModule can fail before a test starts
      self.records.append(make_record(test.id(), status, detail))
      return
    if self.current["status"] not in FAILURES or status == "error":
      self.current["status"] = status
    if detail:
      self.current["detail"] += ("\n\n" if self.current["detail"] else "") + detail

  def addFailure(self, test, err):
    self.mark(test, "failed", self._exc_info_to_string(err, test))

  def addError(self, test, err):
    self.mark(test, "error", self._exc_info_to_string(err, test))

  def addSkip(self, test, reason):
    self.mark(test, "skipped")

  def addExpectedFailure(self, test, err):
    self.mark(test, "xfailed")

  def addUnexpectedSuccess(self, test):
    self.mark(test, "xpassed", "Test was expected to fail, but passed.")

  def addSubTest(self, test, subtest, err):
    if err:
      status = "failed" if issubclass(err[0], test.failureException) else "error"
      self.mark(test, status, f"{subtest}\n{self._exc_info_to_string(err, test)}")


def flatten(suite):
  for test in suite:
    if isinstance(test, unittest.TestSuite):
      yield from flatten(test)
    else:
      yield test


def module_name(path):
  return ".".join(path.resolve().relative_to(ROOT).with_suffix("").parts)


def collect(targets, keyword):
  use_ignores = not targets
  targets = targets or ["openpilot"]
  loader = unittest.TestLoader()
  tests = []
  errors = []
  names = []
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
      if not keyword or keyword.lower() in test.id().lower():
        tests.append(test)
  return list({test.id(): test for test in tests}.values()), errors


def make_batches(tests, workers):
  fixture_groups = {}
  parallel = []
  for test in tests:
    cls = type(test)
    module = sys.modules[cls.__module__]
    if hasattr(module, "setUpModule") or hasattr(module, "tearDownModule"):
      key = cls.__module__
    elif "setUpClass" in cls.__dict__ or "tearDownClass" in cls.__dict__:
      key = f"{cls.__module__}.{cls.__qualname__}"
    else:
      parallel.append(test.id())
      continue
    fixture_groups.setdefault(key, []).append(test.id())
  size = max(1, math.ceil(len(tests) / (workers * 4)))
  batches = list(fixture_groups.values())
  batches.extend(list(batch) for batch in batched(parallel, size))
  return sorted(batches, key=len, reverse=True)


def run_batch(test_ids, capture_output):
  result = Result(capture_output)
  outside = Capture(capture_output)
  os.chdir(ROOT)
  outside.start()
  try:
    unittest.TestLoader().loadTestsFromNames(test_ids).run(result)
  finally:
    stdout, stderr = outside.stop()
  failures = [item for item in result.records if item["status"] in FAILURES]
  if failures:  # attach class/module fixture output to the first related failure
    failures[0]["stdout"] = stdout + failures[0]["stdout"]
    failures[0]["stderr"] = stderr + failures[0]["stderr"]
  return result.records


def run_parallel(batches, workers, warning_action, capture_output):
  with ProcessPoolExecutor(max_workers=workers, initializer=warnings.simplefilter, initargs=(warning_action,)) as pool:
    futures = {pool.submit(run_batch, batch, capture_output): batch for batch in batches}
    for future in as_completed(futures):
      try:
        yield future.result()
      except Exception:
        yield [make_record(futures[future][0], "error", traceback.format_exc())]


def report(records, errors, duration_count, elapsed):
  width = min(100, os.get_terminal_size().columns if sys.stdout.isatty() else 80)
  for index, error in enumerate(errors, 1):
    print(paint(f"\n{'=' * 8} COLLECTION ERROR {index} {'=' * 8}", 31))
    print(error.rstrip())
  for item in sorted((r for r in records if r["status"] in FAILURES), key=lambda r: r["id"]):
    heading = f" {item['status'].upper()} {item['id']} "
    print(paint(f"\n{heading:=^{width}}", 31))
    if item["detail"]:
      print(item["detail"].rstrip())
    for stream in ("stdout", "stderr"):
      if item[stream]:
        print(paint(f"\n--- captured {stream} ---", 33))
        print(item[stream].rstrip())
  timed = sorted((r for r in records if r["time"]), key=lambda r: r["time"], reverse=True)
  if duration_count:
    timed = timed[:duration_count]
  if timed:
    print(paint("\nslowest tests", 36))
    for item in timed:
      print(f"{item['time']:8.2f}s  {item['id']}")
  counts = Counter(item["status"] for item in records)
  parts = [f"{counts[name]} {name}" for name in STATUS_MARKS if counts[name]]
  if errors:
    parts.append(f"{len(errors)} collection error{'s' if len(errors) != 1 else ''}")
  failed = bool(errors) or any(counts[name] for name in FAILURES)
  print(paint(f"\n{', '.join(parts) or 'no tests ran'} in {elapsed:.2f}s", 31 if failed else 32))
  if failed:
    return 1
  if records:
    return 0
  return 5


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("targets", nargs="*", help="files, directories, dotted IDs, or path.py::Class::test")
  parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1, help="workers (default: available CPUs)")
  parser.add_argument("-k", metavar="TEXT", help="only run test IDs containing TEXT")
  parser.add_argument("-s", "--no-capture", action="store_true", help="show test output live")
  parser.add_argument("-v", "--verbose", action="store_true", help="show every test")
  parser.add_argument("--durations", type=int, default=10, metavar="N", help="show N slowest tests; 0 shows all")
  parser.add_argument("-W", "--warnings", choices=("error", "default", "always", "ignore"), default="error")
  args = parser.parse_args()

  capture_output = not args.no_capture
  os.chdir(ROOT)
  warnings.simplefilter(args.warnings)
  started = time.monotonic()
  tests, errors = collect(args.targets, args.k)
  batches = make_batches(tests, args.jobs)
  workers = min(args.jobs, len(batches))
  summary = f"collected {len(tests)} test{'s' if len(tests) != 1 else ''} in {time.monotonic() - started:.2f}s "
  summary += f"• {workers} worker{'s' if workers != 1 else ''}"
  print(summary)
  records = []
  column = 0
  try:
    if workers < 2:
      streams = (run_batch(batch, capture_output) for batch in batches)
    else:
      streams = run_parallel(batches, workers, args.warnings, capture_output)
    for batch in streams:
      records.extend(batch)
      for item in batch:
        mark, code = STATUS_MARKS[item["status"]]
        if args.verbose:
          print(f"{paint(mark, code)} {item['id']} {item['time']:.2f}s")
        else:
          print(paint(mark, code), end="", flush=True)
          column += 1
          if column == 80:
            print()
            column = 0
  except KeyboardInterrupt:
    print(paint("\ninterrupted", 31))
    return 2
  if column:
    print()
  return report(records, errors, args.durations, time.monotonic() - started)


if __name__ == "__main__":
  raise SystemExit(main())
