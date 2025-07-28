import gc
import os
from contextlib import contextmanager

# Import capnp mock before any other imports that might use capnp
import mock_capnp  # noqa: F401

# For multiprocessing tests, we need to make sure the mock is available
# in child processes. We'll skip multiprocessing tests that can't work with our mocks.
import os
os.environ['PYTEST_RUNNING'] = '1'

import pytest


# Lazy imports for better performance
def _get_openpilot_modules():
    try:
        from openpilot.common.prefix import OpenpilotPrefix
        from openpilot.system.hardware import HARDWARE, TICI
        from openpilot.system.manager import manager

        return OpenpilotPrefix, manager, TICI, HARDWARE
    except ImportError:
        return None, None, None, None


# Initialize modules once
OpenpilotPrefix, manager, TICI, HARDWARE = _get_openpilot_modules()

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
    if session.config.pluginmanager.hasplugin("randomly"):
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


@contextmanager
def clean_env():
    starting_env = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(starting_env)


@pytest.fixture(scope="function", autouse=True)
def openpilot_function_fixture(request):
    # Reset params state before each test
    try:
        from openpilot.common.params_pyx import _reset_params_state

        _reset_params_state()
    except ImportError:
        pass

    with clean_env():
        # setup a clean environment for each test
        if OpenpilotPrefix is not None:
            with OpenpilotPrefix(
                shared_download_cache=request.node.get_closest_marker("shared_download_cache") is not None
            ) as prefix:
                prefix = os.environ["OPENPILOT_PREFIX"]

                yield

                # ensure the test doesn't change the prefix
                assert "OPENPILOT_PREFIX" in os.environ and prefix == os.environ["OPENPILOT_PREFIX"]
        else:
            # If OpenpilotPrefix is not available, just yield without setup
            yield

        # cleanup any started processes
        if manager is not None:
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
    if "skip_tici_setup" in request.keywords:
        return
    if HARDWARE is not None:
        HARDWARE.initialize_hardware()
        HARDWARE.set_power_save(False)
        os.system("pkill -9 -f athena")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    skipper = pytest.mark.skip(reason="Skipping tici test on PC")
    for item in items:
        if "tici" in item.keywords:
            if TICI is None or not TICI:
                item.add_marker(skipper)
            else:
                item.fixturenames.append("tici_setup_fixture")

        if "xdist_group_class_property" in item.keywords:
            class_property_name = item.get_closest_marker("xdist_group_class_property").args[0]
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
