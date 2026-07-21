"""Run unittest suites in parallel.

unittest owns argument parsing and discovery. This runner only groups the
resulting suite and executes those groups in non-daemon worker processes.
Set UNITTEST_JOBS (0 means all CPUs) and UNITTEST_LEVEL (module or class) to
configure parallelism.
"""

import concurrent.futures
import io
import multiprocessing
import os
import time
import unittest
import warnings
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
  errors: int
  failures: int
  skipped: int
  expected_failures: int
  unexpected_successes: int
  duration: float

  @property
  def successful(self):
    return not (self.errors or self.failures or self.unexpected_successes)


class _ResultStream(io.StringIO):
  def writeln(self, value=""):
    self.write(f"{value}\n")


@dataclass(frozen=True)
class TestSettings:
  verbosity: int
  failfast: bool
  buffer: bool
  warnings: str | None
  tb_locals: bool


def _run_suite(label, suite, settings):
  stream = _ResultStream()
  result = unittest.TextTestResult(stream, descriptions=True, verbosity=settings.verbosity)
  result.failfast = settings.failfast
  result.buffer = settings.buffer
  result.tb_locals = settings.tb_locals

  start_time = time.perf_counter()
  with warnings.catch_warnings():
    if settings.warnings:
      warnings.simplefilter(settings.warnings)
    result.startTestRun()
    try:
      suite(result)
    finally:
      result.stopTestRun()
  if not result.wasSuccessful():
    result.printErrors()

  return TaskResult(
    label,
    stream.getvalue(),
    result.testsRun,
    len(result.errors),
    len(result.failures),
    len(result.skipped),
    len(result.expectedFailures),
    len(result.unexpectedSuccesses),
    time.perf_counter() - start_time,
  )


def _run_task(task, settings):
  suite = unittest.defaultTestLoader.loadTestsFromNames(task.test_ids)
  return _run_suite(task.label, suite, settings)


def _init_worker(test_start_method):
  # Spawning isolates workers, while restoring the platform default here lets
  # tests create child processes exactly as they do under plain unittest.
  multiprocessing.set_start_method(test_start_method, force=True)


def _iter_test_cases(suite):
  if isinstance(suite, unittest.TestCase):
    yield suite
  else:
    for test in suite:
      yield from _iter_test_cases(test)


def _make_tasks(suite, level):
  grouped = {}
  excluded = 0
  failed_tests = []

  for test in _iter_test_cases(suite):
    test_class = type(test)
    if test_class.__module__ == "unittest.loader" and test_class.__name__ == "_FailedTest":
      failed_tests.append(test)
    elif os.environ.get("SKIP_SLOW") and getattr(test_class, "SLOW_TEST", False):
      excluded += 1
    else:
      module = test_class.__module__
      key = f"{module}.{test_class.__qualname__}" if level == "class" else module
      grouped.setdefault(key, []).append(test.id())

  tasks = [TestTask(label, tuple(test_ids)) for label, test_ids in grouped.items()]
  tasks.sort(key=lambda task: len(task.test_ids), reverse=True)
  return tasks, excluded, failed_tests


class ParallelTestRunner(unittest.TextTestRunner):
  def run(self, test):
    jobs = int(os.environ.get("UNITTEST_JOBS", "0"))
    level = os.environ.get("UNITTEST_LEVEL", "module")
    if jobs < 0:
      raise ValueError("UNITTEST_JOBS must be non-negative")
    if level not in ("module", "class"):
      raise ValueError("UNITTEST_LEVEL must be 'module' or 'class'")

    tasks, excluded, failed_tests = _make_tasks(test, level)
    worker_count = min(len(tasks), jobs or (os.cpu_count() or 1))
    self.stream.writeln(f"Discovered {test.countTestCases()} tests; running {len(tasks)} {level} suites across {worker_count} workers")

    settings = TestSettings(self.verbosity, self.failfast, self.buffer, self.warnings, self.tb_locals)
    result = self._makeResult()
    unittest.registerResult(result)
    results = []
    worker_errors = []
    start_time = time.perf_counter()

    if failed_tests:
      collection_result = _run_suite("collection errors", unittest.TestSuite(failed_tests), settings)
      results.append(collection_result)
      self.stream.write(collection_result.output)

    stop_early = self.failfast and any(not result.successful for result in results)
    if tasks and not stop_early:
      executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
        initargs=(multiprocessing.get_start_method(),),
      )
      futures = {executor.submit(_run_task, task, settings): task for task in tasks}
      try:
        for future in concurrent.futures.as_completed(futures):
          task = futures[future]
          try:
            task_result = future.result()
          except Exception as exc:
            worker_errors.append(f"Worker failed while running {task.label}: {exc!r}")
            stop_early = self.failfast
          else:
            results.append(task_result)
            if task_result.output:
              self.stream.write(task_result.output)
              self.stream.flush()
            stop_early = self.failfast and not task_result.successful

          stop_early = stop_early or result.shouldStop
          if stop_early:
            for pending in futures:
              pending.cancel()
            break
      finally:
        executor.shutdown(wait=True, cancel_futures=stop_early)

    duration = time.perf_counter() - start_time
    if self.verbosity and results:
      self.stream.writeln()

    if self.durations is not None and results:
      duration_count = len(results) if self.durations == 0 else self.durations
      self.stream.writeln("Slowest test suites:")
      for task_result in sorted(results, key=lambda item: item.duration, reverse=True)[:duration_count]:
        self.stream.writeln(f"{task_result.duration:8.3f}s  {task_result.label} ({task_result.tests_run} tests)")
      self.stream.writeln()

    for error in worker_errors:
      self.stream.writeln(error)

    result.testsRun = sum(item.tests_run for item in results)
    result.errors = [(None, "")] * (sum(item.errors for item in results) + len(worker_errors))
    result.failures = [(None, "")] * sum(item.failures for item in results)
    result.skipped = [(None, "")] * sum(item.skipped for item in results)
    result.expectedFailures = [(None, "")] * sum(item.expected_failures for item in results)
    result.unexpectedSuccesses = [None] * sum(item.unexpected_successes for item in results)

    infos = []
    for values, label in (
      (result.failures, "failures"),
      (result.errors, "errors"),
      (result.skipped, "skipped"),
      (result.expectedFailures, "expected failures"),
      (result.unexpectedSuccesses, "unexpected successes"),
    ):
      if values:
        infos.append(f"{label}={len(values)}")
    if excluded:
      infos.append(f"excluded={excluded}")

    self.stream.writeln(unittest.TextTestResult.separator2)
    self.stream.writeln(f"Ran {result.testsRun} {'tests' if result.testsRun != 1 else 'test'} in {duration:.3f}s")
    self.stream.writeln()
    suffix = f" ({', '.join(infos)})" if infos else ""
    self.stream.writeln(f"{'OK' if result.wasSuccessful() else 'FAILED'}{suffix}")
    return result


if __name__ == "__main__":
  unittest.main(module=None, testRunner=ParallelTestRunner)
