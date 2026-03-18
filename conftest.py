import contextlib
import gc
import os
import pytest

# Imports moved to lazy loading to speed up pytest --collect-only
# See: https://github.com/commaai/openpilot/issues/32611

# Lazy import helpers - only load hardware/system modules when actually needed
def _get_openpilot_prefix():
  from openpilot.common.prefix import OpenpilotPrefix
  return OpenpilotPrefix

def _get_manager():
  from openpilot.system.manager import manager
  return manager

def _get_hardware():
  from openpilot.system.hardware import HARDWARE
  return HARDWARE

def _get_tici():
  from openpilot.system.hardware import TICI
  return TICI

# TODO: pytest-cpp doesn't support FAIL, and we need to create test translations in sessionstart
# pending https://github.com/pytest-dev/pytest-cpp/pull/147
collect_ignore = [
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
  HARDWARE = _get_hardware()
  if 'skip_tici_setup' in request.keywords:
    return
  HARDWARE.initialize_hardware()
  HARDWARE.set_power_save(False)
  os.system("pkill -9 -f athena")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
  skipper = pytest.mark.skip(reason="Skipping tici test on PC")
  TICI = _get_tici()
  for item in items:
    if "tici" in item.keywords:
      if not TICI:
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

  # OPTIMIZATION: Speed up pytest --collect-only by limiting hypothesis examples
  # See: https://github.com/commaai/openpilot/issues/32611
  if config.getoption('--collect-only'):
    try:
      from hypothesis import settings, HealthCheck
      settings.register_profile("collect", max_examples=1, suppress_health_check=[HealthCheck.too_slow])
      settings.load_profile("collect")
    except ImportError:
      pass  # hypothesis not installed
