"""A small, unittest-compatible parallel test runner.

Tests are collected once in the controller and run as module suites in spawned,
non-daemon workers. This preserves unittest fixtures and lets tests create their
own child processes without making every worker collect the entire repository.
"""

import argparse
import concurrent.futures
import io
import multiprocessing
import os
from pathlib import Path
import sys
import time
import traceback
import unittest
import warnings
from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class TestTask:
  label: str
  test_ids: tuple[str, ...]


@dataclass(frozen=True)
class TaskResult:
  label: str
  output: str
  tests_run: int
  errors: tuple[str, ...]
  failures: tuple[str, ...]
  skipped: int
  expected_failures: int
  unexpected_successes: tuple[str, ...]
  duration: float

  @property
  def successful(self) -> bool:
    return not (self.errors or self.failures or self.unexpected_successes)


class _ResultStream(io.StringIO):
  def writeln(self, value=""):
    self.write(f"{value}\n")


def _format_error(result, kind, error):
  test, traceback = error
  return "\n".join(
    (
      unittest.TextTestResult.separator1,
      f"{kind}: {result.getDescription(test)}",
      unittest.TextTestResult.separator2,
      traceback,
    )
  )


def _run_task(task, verbosity, failfast, buffer):
  """Load and run one isolated module or class suite in a worker process."""
  loader = unittest.TestLoader()
  suite = loader.loadTestsFromNames(task.test_ids)
  stream = _ResultStream()
  result = unittest.TextTestResult(stream, descriptions=True, verbosity=verbosity)
  result.failfast = failfast
  result.buffer = buffer

  start_time = time.perf_counter()
  result.startTestRun()
  try:
    suite(result)
  finally:
    result.stopTestRun()

  return TaskResult(
    task.label,
    stream.getvalue(),
    result.testsRun,
    tuple(_format_error(result, "ERROR", error) for error in result.errors),
    tuple(_format_error(result, "FAIL", failure) for failure in result.failures),
    len(result.skipped),
    len(result.expectedFailures),
    tuple(result.getDescription(test) for test in result.unexpectedSuccesses),
    time.perf_counter() - start_time,
  )


def _init_worker(test_start_method, warning_action):
  # ProcessPoolExecutor's spawn context otherwise leaks into tests. Restore
  # the platform default so nested multiprocessing behaves like plain unittest.
  multiprocessing.set_start_method(test_start_method, force=True)
  if warning_action:
    warnings.simplefilter(warning_action)


def _iter_test_cases(suite):
  if isinstance(suite, unittest.TestCase):
    yield suite
  else:
    for test in suite:
      yield from _iter_test_cases(test)


def _task_key(test, level):
  test_class = type(test)
  module = test_class.__module__
  if level == "test":
    return test.id()
  if level == "class":
    return f"{module}.{test_class.__qualname__}"
  return module


def _make_tasks(suite, level, skip_slow=False):
  grouped = OrderedDict()
  excluded = 0
  failed_tests = []

  for test in _iter_test_cases(suite):
    if type(test).__module__ == "unittest.loader" and type(test).__name__ == "_FailedTest":
      failed_tests.append(test)
      continue
    if skip_slow and getattr(type(test), "SLOW_TEST", False):
      excluded += 1
      continue

    key = _task_key(test, level)
    grouped.setdefault(key, []).append(test.id())

  tasks = [TestTask(label, tuple(test_ids)) for label, test_ids in grouped.items()]
  # Starting larger suites first is a simple approximation of longest-job-first
  # scheduling and reduces the chance that one large module becomes the tail.
  tasks.sort(key=lambda task: len(task.test_ids), reverse=True)
  return tasks, excluded, failed_tests


def _convert_test_name(name, top_level):
  path = Path(name)
  if path.suffix == ".py" and path.exists():
    top = Path(top_level or ".").resolve()
    relative = path.resolve().relative_to(top)
    return ".".join(relative.with_suffix("").parts)
  return name


def _select_pattern(pattern):
  return pattern if "*" in pattern else f"*{pattern}*"


def _add_namespace_tests(loader, suite, start_directory, top_level_directory, pattern):
  """Add tests below namespace directories that unittest discovery skips."""
  start = Path(start_directory).resolve()
  top = Path(top_level_directory or start_directory).resolve()
  existing_modules = set()
  for test in _iter_test_cases(suite):
    if type(test).__module__ == "unittest.loader" and type(test).__name__ == "_FailedTest":
      existing_modules.add(test._testMethodName)
    else:
      existing_modules.add(type(test).__module__)
  extra_suites = []

  for test_file in sorted(start.rglob(pattern)):
    try:
      relative = test_file.relative_to(top)
    except ValueError:
      continue
    module = ".".join(relative.with_suffix("").parts)
    if module not in existing_modules:
      try:
        extra_suites.append(loader.loadTestsFromName(module))
      except Exception:
        loader.errors.append(f"Failed to import test module: {module}\n{traceback.format_exc()}")
      existing_modules.add(module)

  return unittest.TestSuite((suite, *extra_suites))


def _create_parser():
  parser = argparse.ArgumentParser(prog="parallel-unittest")
  parser.add_argument("tests", nargs="*", help="Test modules, classes, methods, or Python files")
  parser.add_argument("-v", "--verbose", dest="verbosity", action="store_const", const=2, default=1)
  parser.add_argument("-q", "--quiet", dest="verbosity", action="store_const", const=0)
  parser.add_argument("-f", "--failfast", action="store_true")
  parser.add_argument("-b", "--buffer", action="store_true")
  parser.add_argument("-k", dest="patterns", action="append", default=[], help="Only run tests matching this pattern")
  parser.add_argument("-s", "--start-directory", default=".")
  parser.add_argument("-p", "--pattern", default="test_*.py")
  parser.add_argument("-t", "--top-level-directory")
  parser.add_argument("-j", "--jobs", type=int, default=0, help="Worker count; 0 uses all logical CPUs")
  parser.add_argument(
    "--level",
    choices=("module", "class", "test"),
    default="module",
    help="Parallelize modules, classes, or individual tests; finer levels repeat outer fixtures",
  )
  parser.add_argument("--durations", type=int, default=10, help="Show the N slowest suites; 0 shows all")
  parser.add_argument("-W", "--warnings", choices=("error", "default", "always", "ignore", "module", "once"))
  return parser


def _load_tests(args):
  loader = unittest.TestLoader()
  if args.patterns:
    loader.testNamePatterns = [_select_pattern(pattern) for pattern in args.patterns]

  if args.tests:
    names = [_convert_test_name(name, args.top_level_directory) for name in args.tests]
    suite = loader.loadTestsFromNames(names)
  else:
    suite = loader.discover(args.start_directory, pattern=args.pattern, top_level_dir=args.top_level_directory)
    suite = _add_namespace_tests(loader, suite, args.start_directory, args.top_level_directory, args.pattern)
  return loader, suite


def _print_summary(tests_run, duration, failures, errors, skipped, expected_failures, unexpected_successes, excluded):
  infos = []
  if failures:
    infos.append(f"failures={failures}")
  if errors:
    infos.append(f"errors={errors}")
  if skipped:
    infos.append(f"skipped={skipped}")
  if expected_failures:
    infos.append(f"expected failures={expected_failures}")
  if unexpected_successes:
    infos.append(f"unexpected successes={unexpected_successes}")
  if excluded:
    infos.append(f"excluded={excluded}")

  successful = not (failures or errors or unexpected_successes)
  print(unittest.TextTestResult.separator2, file=sys.stderr)
  print(f"Ran {tests_run} {'tests' if tests_run != 1 else 'test'} in {duration:.3f}s", file=sys.stderr)
  print(file=sys.stderr)
  suffix = f" ({', '.join(infos)})" if infos else ""
  print(f"{'OK' if successful else 'FAILED'}{suffix}", file=sys.stderr)
  return successful


def main(argv=None):
  parser = _create_parser()
  args = parser.parse_args(argv)
  if args.jobs < 0:
    parser.error("--jobs must be non-negative")
  if args.durations < 0:
    parser.error("--durations must be non-negative")
  if args.warnings:
    warnings.simplefilter(args.warnings)

  collection_start = time.perf_counter()
  loader, suite = _load_tests(args)
  skip_slow = bool(os.environ.get("SKIP_SLOW"))
  tasks, excluded, failed_tests = _make_tasks(suite, args.level, skip_slow=skip_slow)
  collection_duration = time.perf_counter() - collection_start

  collection_errors = list(loader.errors)

  worker_count = min(len(tasks), args.jobs or (os.cpu_count() or 1))
  total_discovered = sum(len(task.test_ids) for task in tasks) + excluded + len(failed_tests)
  discovery_report = "".join(
    (
      f"Discovered {total_discovered} tests in {collection_duration:.3f}s; ",
      f"running {len(tasks)} {args.level} suites across {worker_count or 0} workers",
    )
  )
  print(discovery_report, file=sys.stderr)

  results = []
  worker_errors = []
  start_time = time.perf_counter()
  if tasks:
    test_start_method = multiprocessing.get_start_method()
    context = multiprocessing.get_context("spawn")
    executor = concurrent.futures.ProcessPoolExecutor(
      max_workers=worker_count,
      mp_context=context,
      initializer=_init_worker,
      initargs=(test_start_method, args.warnings),
    )
    futures = {executor.submit(_run_task, task, args.verbosity, args.failfast, args.buffer): task for task in tasks}
    stop_early = False
    try:
      for future in concurrent.futures.as_completed(futures):
        task = futures[future]
        try:
          result = future.result()
        except Exception as exc:
          worker_errors.append(f"Worker failed while running {task.label}: {exc!r}")
          stop_early = args.failfast
        else:
          results.append(result)
          if result.output:
            print(result.output, end="", file=sys.stderr, flush=True)
          stop_early = args.failfast and not result.successful

        if stop_early:
          for pending in futures:
            pending.cancel()
          break
    finally:
      executor.shutdown(wait=True, cancel_futures=stop_early)
  duration = time.perf_counter() - start_time

  if args.verbosity and results:
    print(file=sys.stderr)

  duration_count = len(results) if args.durations == 0 else args.durations
  if duration_count and results:
    print("Slowest test suites:", file=sys.stderr)
    for result in sorted(results, key=lambda item: item.duration, reverse=True)[:duration_count]:
      print(f"{result.duration:8.3f}s  {result.label} ({result.tests_run} tests)", file=sys.stderr)
    print(file=sys.stderr)

  errors = [error for result in results for error in result.errors]
  failures = [failure for result in results for failure in result.failures]
  unexpected_successes = [test for result in results for test in result.unexpected_successes]
  all_errors = collection_errors + worker_errors + errors
  for error in all_errors:
    print(error, file=sys.stderr)
  for failure in failures:
    print(failure, file=sys.stderr)
  for test in unexpected_successes:
    print(f"UNEXPECTED SUCCESS: {test}", file=sys.stderr)

  successful = _print_summary(
    sum(result.tests_run for result in results),
    duration,
    len(failures),
    len(all_errors),
    sum(result.skipped for result in results),
    sum(result.expected_failures for result in results),
    len(unexpected_successes),
    excluded,
  )
  return 0 if successful else 1


if __name__ == "__main__":
  raise SystemExit(main())
