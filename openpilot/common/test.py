import contextlib
import gc
import inspect
import os
import subprocess
import unittest
from unittest import mock

from openpilot.common.hardware import HARDWARE, TICI
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.system.manager import manager


@contextlib.contextmanager
def clean_env():
  starting_env = dict(os.environ)
  try:
    yield
  finally:
    os.environ.clear()
    os.environ.update(starting_env)


class OpenpilotTestCase(unittest.TestCase):
  """TestCase with openpilot's per-test isolation and legacy hook support."""

  TICI_TEST = False
  SKIP_TICI_SETUP = False
  SHARED_DOWNLOAD_CACHE = False
  SLOW_TEST = False

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    if cls.SLOW_TEST and os.environ.get("SKIP_SLOW"):
      cls.__unittest_skip__ = True
      cls.__unittest_skip_why__ = "slow test"
    elif cls.TICI_TEST and not TICI:
      cls.__unittest_skip__ = True
      cls.__unittest_skip_why__ = "Skipping tici test on PC"

    # Preserve legacy xunit hook names and invoke them inside the
    # OpenpilotPrefix boundary below.
    for name in ("setup_method", "teardown_method"):
      hook = cls.__dict__.get(name)
      if hook is not None:
        setattr(cls, f"openpilot_{name}", hook)
        setattr(cls, name, None)

  def _fixture(self, name):
    if name == "mocker":
      return Mocker(self.addCleanup)
    if name == "monkeypatch":
      return MonkeyPatch(self.addCleanup)
    if name == "subtests":
      return SubTests(self)

    fixture = getattr(inspect.getmodule(type(self)), name)
    kwargs = {p: self._fixture(p) for p in inspect.signature(fixture).parameters}
    value = fixture(**kwargs)
    if inspect.isgenerator(value):
      generator = value
      value = next(generator)
      self.addCleanup(lambda: next(generator, None))
    return value

  def _callTestMethod(self, method):
    params = [name for name, param in inspect.signature(method).parameters.items()
              if param.default is inspect.Parameter.empty]
    return method(**{name: self._fixture(name) for name in params})

  def run(self, result=None):
    # This boundary cannot live in setUp/tearDown: existing unittest classes
    # are allowed to override those hooks without calling super().
    if ((self.SLOW_TEST and os.environ.get("SKIP_SLOW")) or
        (self.TICI_TEST and not TICI) or getattr(type(self), "__unittest_skip__", False)):
      return super().run(result)
    test_env = clean_env()
    test_env.__enter__()
    prefix = OpenpilotPrefix(shared_download_cache=self.SHARED_DOWNLOAD_CACHE)
    prefix.__enter__()
    try:
      return super().run(result)
    finally:
      prefix.__exit__(None, None, None)
      manager.manager_cleanup()
      if not gc.isenabled():
        gc.enable()
        gc.collect()
      test_env.__exit__(None, None, None)

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    if cls.SLOW_TEST and os.environ.get("SKIP_SLOW"):
      raise unittest.SkipTest("slow test")
    if cls.TICI_TEST and not TICI:
      raise unittest.SkipTest("Skipping tici test on PC")
    cls._class_env = clean_env()
    cls._class_env.__enter__()
    setup_class = getattr(cls, "setup_class", None)
    if setup_class is not None:
      setup_class()

  @classmethod
  def tearDownClass(cls):
    try:
      teardown_class = getattr(cls, "teardown_class", None)
      if teardown_class is not None:
        teardown_class()
    finally:
      cls._class_env.__exit__(None, None, None)
      super().tearDownClass()

  def setUp(self):
    super().setUp()
    if self.TICI_TEST and not TICI:
      self.skipTest("Skipping tici test on PC")

    if self.TICI_TEST and not self.SKIP_TICI_SETUP:
      HARDWARE.initialize_hardware()
      HARDWARE.set_power_save(False)
      subprocess.run(["pkill", "-9", "-f", "athena"], check=False)

    setup_method = getattr(self, "openpilot_setup_method", None)
    if setup_method is not None:
      setup_method()

  def tearDown(self):
    try:
      teardown_method = getattr(self, "openpilot_teardown_method", None)
      if teardown_method is not None:
        teardown_method()
    finally:
      super().tearDown()


class Mocker:
  Mock = mock.Mock
  MagicMock = mock.MagicMock
  call = mock.call
  ANY = mock.ANY

  def __init__(self, add_cleanup):
    self._add_cleanup = add_cleanup
    self.patch = Patch(self._start)

  def _start(self, patcher):
    value = patcher.start()
    self._add_cleanup(patcher.stop)
    return value


class Patch:
  def __init__(self, start):
    self._start = start

  def __call__(self, *args, **kwargs):
    return self._start(mock.patch(*args, **kwargs))

  def object(self, *args, **kwargs):
    return self._start(mock.patch.object(*args, **kwargs))


class MonkeyPatch:
  def __init__(self, add_cleanup):
    self._add_cleanup = add_cleanup

  def setattr(self, target, name, value):
    patcher = mock.patch.object(target, name, value)
    patcher.start()
    self._add_cleanup(patcher.stop)


class SubTests:
  def __init__(self, test_case):
    self._test_case = test_case

  def test(self, label=None, **kwargs):
    return self._test_case.subTest(**kwargs) if label is None else self._test_case.subTest(label, **kwargs)
