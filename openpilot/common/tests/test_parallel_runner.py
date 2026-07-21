import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

from openpilot.common.test import OpenpilotTestCase
from openpilot.common.parameterized import parameterized_class
from openpilot.common.parallel_runner import _make_tasks


class TestParallelRunner(OpenpilotTestCase):
  def test_parameterized_class_skip_state(self):
    class Regular(unittest.TestCase):
      def test_example(self):
        pass

    regular_name = f"{Regular.__name__}_0"
    parameterized_class(("value",), [(1,)])(Regular)
    self.addCleanup(globals().pop, regular_name)
    assert not globals()[regular_name].__unittest_skip__

    old_skip_slow = os.environ.get("SKIP_SLOW")
    os.environ["SKIP_SLOW"] = "1"
    try:

      class Slow(OpenpilotTestCase):
        SLOW_TEST = True

        def test_example(self):
          pass

      slow_name = f"{Slow.__name__}_0"
      parameterized_class(("value",), [(1,)])(Slow)
    finally:
      if old_skip_slow is None:
        os.environ.pop("SKIP_SLOW")
      else:
        os.environ["SKIP_SLOW"] = old_skip_slow

    self.addCleanup(globals().pop, slow_name)
    assert globals()[slow_name].__unittest_skip__
    assert globals()[slow_name].__unittest_skip_why__ == "slow test"

  def test_task_grouping_and_slow_filter(self):
    class ExampleOne(unittest.TestCase):
      def test_one(self):
        pass

    class ExampleTwo(unittest.TestCase):
      SLOW_TEST = True

      def test_two(self):
        pass

    suite = unittest.TestSuite((ExampleOne("test_one"), ExampleTwo("test_two")))

    module_tasks, excluded, _ = _make_tasks(suite, "module", skip_slow=True)
    assert len(module_tasks) == 1
    assert module_tasks[0].test_ids == (ExampleOne("test_one").id(),)
    assert excluded == 1

    class_tasks, excluded, _ = _make_tasks(suite, "class")
    assert len(class_tasks) == 2
    assert excluded == 0

    test_tasks, excluded, _ = _make_tasks(suite, "test")
    assert len(test_tasks) == 2
    assert excluded == 0

  def _run_fixture(self, source, namespace_source=None):
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      tests_dir = root / "tests"
      tests_dir.mkdir()
      (tests_dir / "__init__.py").write_text("")
      (tests_dir / "test_fixture.py").write_text(textwrap.dedent(source))
      if namespace_source is not None:
        namespace_dir = tests_dir / "namespace"
        namespace_dir.mkdir()
        (namespace_dir / "test_namespace.py").write_text(textwrap.dedent(namespace_source))

      env = os.environ.copy()
      env["PYTHONPATH"] = os.pathsep.join((str(root), os.getcwd()))
      return subprocess.run(
        [sys.executable, "-m", "openpilot.common.parallel_runner", "-j", "2", "-s", str(tests_dir), "-t", str(root), "--durations", "0"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
      ), root

  def test_unittest_fixtures_skips_expected_failures_and_child_processes(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      marker = Path(temp_dir) / "fixtures"
      result, _ = self._run_fixture(
        f"""
        import multiprocessing
        from pathlib import Path
        import unittest

        MARKER = Path({str(marker)!r})

        def mark(value):
          with MARKER.open("a") as f:
            f.write(value + "\\n")

        def child(queue):
          queue.put("child")

        def setUpModule():
          mark("module setup")

        def tearDownModule():
          mark("module teardown")

        class TestFixture(unittest.TestCase):
          @classmethod
          def setUpClass(cls):
            mark("class setup")

          @classmethod
          def tearDownClass(cls):
            mark("class teardown")

          def test_child_process(self):
            queue = multiprocessing.Queue()
            process = multiprocessing.Process(target=child, args=(queue,))
            process.start()
            process.join(5)
            self.assertEqual(process.exitcode, 0)
            self.assertEqual(queue.get(timeout=1), "child")

          def test_success(self):
            self.assertTrue(True)

          @unittest.skip("example skip")
          def test_skip(self):
            self.fail()

          @unittest.expectedFailure
          def test_expected_failure(self):
            self.fail("expected")
      """,
        """
        import unittest

        class TestNamespaceDirectory(unittest.TestCase):
          def test_discovered(self):
            self.assertTrue(True)
      """,
      )

      assert result.returncode == 0, result.stderr
      assert "Ran 5 tests" in result.stderr
      assert "skipped=1" in result.stderr
      assert "expected failures=1" in result.stderr
      assert marker.read_text().splitlines() == ["module setup", "class setup", "class teardown", "module teardown"]

  def test_failure_and_collection_error_reporting(self):
    failure, _ = self._run_fixture("""
      import unittest

      class TestFailure(unittest.TestCase):
        def test_failure(self):
          self.assertEqual(1, 2)
    """)
    assert failure.returncode == 1
    assert "FAIL: test_failure" in failure.stderr
    assert "FAILED (failures=1)" in failure.stderr

    collection_error, _ = self._run_fixture("this is not valid Python")
    assert collection_error.returncode == 1
    assert "Failed to import test module" in collection_error.stderr
    assert "FAILED (errors=1)" in collection_error.stderr
