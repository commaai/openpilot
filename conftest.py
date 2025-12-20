import contextlib
import gc
import os

# Pre-warm heavy imports before pytest collection
try:
    import numpy  # noqa: F401
except ImportError:
    pass

try:
    import hypothesis  # noqa: F401
except ImportError:
    pass

try:
    import cereal
except ImportError:
    pass

try:
    from opendbc import car  # noqa: F401
except ImportError:
    pass

try:
    from openpilot.tools.lib.logreader import LogReader  # noqa: F401
except ImportError:
    pass

try:
    import casadi  # noqa: F401
except ImportError:
    pass

try:
    from openpilot.common.params import Params  # noqa: F401
except ImportError:
    pass

try:
    from openpilot.selfdrive.test import helpers  # noqa: F401
except ImportError:
    pass

try:
    import parameterized  # noqa: F401
except ImportError:
    pass

try:
    import cereal.messaging  # noqa: F401
except ImportError:
    pass

try:
    from cereal import log  # noqa: F401
except ImportError:
    pass

try:
    from panda import Panda  # noqa: F401
except ImportError:
    pass

try:
    from openpilot.system.manager.process_config import managed_processes  # noqa: F401
except ImportError:
    pass

import pytest
from typing import Any

# Lazy-loaded modules for fixtures
_OpenpilotPrefix: Any = None
_manager: Any = None
_HARDWARE: Any = None


def _get_openpilot_prefix():
    global _OpenpilotPrefix
    if _OpenpilotPrefix is None:
        from openpilot.common.prefix import OpenpilotPrefix
        _OpenpilotPrefix = OpenpilotPrefix
    return _OpenpilotPrefix


def _get_manager():
    global _manager
    if _manager is None:
        from openpilot.system.manager import manager
        _manager = manager
    return _manager


def _get_hardware():
    global _HARDWARE
    if _HARDWARE is None:
        from openpilot.system.hardware import HARDWARE
        _HARDWARE = HARDWARE
    return _HARDWARE


def _is_tici():
    return os.path.isfile('/TICI')


# TODO: pytest-cpp doesn't support FAIL, and we need to create test translations in sessionstart
# pending https://github.com/pytest-dev/pytest-cpp/pull/147
collect_ignore = [
  "selfdrive/ui/tests/test_translations",
  "selfdrive/test/process_replay/test_processes.py",
  "selfdrive/test/process_replay/test_regen.py",
]
collect_ignore_glob = [
  "selfdrive/debug/*.py",
  "selfdrive/modeld/*.py",
]


def pytest_sessionstart(session):
  # TODO: fix tests and enable test order randomization
  if session.config.pluginmanager.hasplugin('randomly'):
    session.config.option.randomly_reorganize = False


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_runtest_call(item):
  # ensure we run as a hook after capturemanager's
  if item.get_closest_marker("nocapture") is not None:
    capmanager = item.config.pluginmanager.getplugin("capturemanager")
    with capmanager.global_and_fixture_disabled():
      yield
  else:
    yield


@contextlib.contextmanager
def clean_env():
  starting_env = dict(os.environ)
  yield
  os.environ.clear()
  os.environ.update(starting_env)


@pytest.fixture(scope="function", autouse=True)
def openpilot_function_fixture(request):
  OpenpilotPrefix = _get_openpilot_prefix()
  manager = _get_manager()

  with clean_env():
    # setup a clean environment for each test
    with OpenpilotPrefix(shared_download_cache=request.node.get_closest_marker("shared_download_cache") is not None) as prefix:
      prefix = os.environ["OPENPILOT_PREFIX"]

      yield

      # ensure the test doesn't change the prefix
      assert "OPENPILOT_PREFIX" in os.environ and prefix == os.environ["OPENPILOT_PREFIX"]

    # cleanup any started processes
    manager.manager_cleanup()

    # some processes disable gc for performance, re-enable here
    if not gc.isenabled():
      gc.enable()
      gc.collect()

# If you use setUpClass, the environment variables won't be cleared properly,
# so we need to hook both the function and class pytest fixtures
@pytest.fixture(scope="class", autouse=True)
def openpilot_class_fixture():
  with clean_env():
    yield


@pytest.fixture(scope="function")
def tici_setup_fixture(request, openpilot_function_fixture):
  """Ensure a consistent state for tests on-device. Needs the openpilot function fixture to run first."""
  if 'skip_tici_setup' in request.keywords:
    return
  HARDWARE = _get_hardware()
  HARDWARE.initialize_hardware()
  HARDWARE.set_power_save(False)
  os.system("pkill -9 -f athena")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
  skipper = pytest.mark.skip(reason="Skipping tici test on PC")
  is_tici = _is_tici()
  for item in items:
    if "tici" in item.keywords:
      if not is_tici:
        item.add_marker(skipper)
      else:
        item.fixturenames.append('tici_setup_fixture')

    if "xdist_group_class_property" in item.keywords:
      class_property_name = item.get_closest_marker('xdist_group_class_property').args[0]
      class_property_value = getattr(item.cls, class_property_name)
      item.add_marker(pytest.mark.xdist_group(class_property_value))


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
  config_line = "xdist_group_class_property: group tests by a property of the class that contains them"
  config.addinivalue_line("markers", config_line)

  config_line = "nocapture: don't capture test output"
  config.addinivalue_line("markers", config_line)

  config_line = "shared_download_cache: share download cache between tests"
  config.addinivalue_line("markers", config_line)
