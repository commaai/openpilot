import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "tools/test_runner.py"


class TestTestRunner(unittest.TestCase):
  def run_runner(self, sources, *args, env=None):
    if isinstance(sources, str):
      sources = {"test_sample.py": sources}
      target_file = True
    else:
      target_file = False

    with tempfile.TemporaryDirectory(prefix="runner_test_", dir=ROOT) as directory:
      directory = Path(directory)
      for name, source in sources.items():
        (directory / name).write_text(textwrap.dedent(source))
      target = directory / next(iter(sources)) if target_file else directory
      return subprocess.run(
        [sys.executable, RUNNER, "--durations", "0", *args, target],
        cwd=ROOT, env=os.environ | (env or {}), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
      )

  def test_parallel_results_and_failure_capture(self):
    result = self.run_runner("""
      import os
      import multiprocessing
      import unittest

      def child():
        pass

      class TestEverything(unittest.TestCase):
        def test_pass(self):
          print("hidden success output")

        def test_child_process(self):
          process = multiprocessing.Process(target=child)
          process.start()
          process.join()
          self.assertEqual(process.exitcode, 0)

        def test_fail(self):
          print("python stdout")
          os.write(1, b"native stdout\\n")
          os.write(2, b"native stderr\\n")
          self.assertEqual(1, 2)

        @unittest.skip("not today")
        def test_skip(self):
          pass

        @unittest.expectedFailure
        def test_expected_failure(self):
          self.fail("expected")

        @unittest.expectedFailure
        def test_unexpected_success(self):
          pass
    """, "-j", "2")

    self.assertEqual(result.returncode, 1, result.stdout)
    for text in ("2 passed", "1 failed", "1 skipped", "1 xfailed", "1 xpassed",
                 "captured stdout", "python stdout", "native stdout",
                 "captured stderr", "native stderr", "AssertionError"):
      assert text in result.stdout
    assert "hidden success output" not in result.stdout

  def test_filter(self):
    result = self.run_runner("""
      import unittest

      class TestSelection(unittest.TestCase):
        def test_selected(self):
          pass

        def test_other(self):
          self.fail()
    """, "-k", "selected")

    self.assertEqual(result.returncode, 0, result.stdout)
    assert "collected 1 test" in result.stdout
    assert "1 passed" in result.stdout
    assert "test_other" not in result.stdout

  def test_collection_errors_do_not_hide_good_tests(self):
    result = self.run_runner({
      "test_bad.py": 'raise RuntimeError("broken import")',
      "test_good.py": """
        import unittest

        class TestGood(unittest.TestCase):
          def test_good(self):
            pass
      """,
    })

    self.assertEqual(result.returncode, 1, result.stdout)
    assert "1 passed" in result.stdout
    assert "1 collection error" in result.stdout
    assert "broken import" in result.stdout

  def test_class_fixture_is_kept_together(self):
    with tempfile.NamedTemporaryFile() as counter:
      result = self.run_runner("""
        import os
        import unittest

        class TestFixture(unittest.TestCase):
          @classmethod
          def setUpClass(cls):
            with open(os.environ["RUNNER_COUNTER"], "a") as stream:
              stream.write("setup\\n")

          def test_one(self):
            pass

          def test_two(self):
            pass
      """, "-j", "2", env={"RUNNER_COUNTER": counter.name})
      counter.seek(0)
      setup_calls = counter.read()

    self.assertEqual(result.returncode, 0, result.stdout)
    self.assertEqual(setup_calls, b"setup\n")

  def test_worker_crash_is_reported(self):
    result = self.run_runner("""
      import os
      import time
      import unittest

      class TestCrash(unittest.TestCase):
        def test_crash(self):
          os._exit(7)

        def test_other(self):
          time.sleep(0.1)
    """, "-j", "2")

    self.assertEqual(result.returncode, 1, result.stdout)
    assert "terminated abruptly" in result.stdout
    assert "error" in result.stdout
