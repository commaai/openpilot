import os
import pytest
import random

from openpilot.common.prefix import OpenpilotPrefix
from openpilot.system.hardware import TICI


@pytest.fixture(scope="function", autouse=True)
def openpilot_function_fixture():
  starting_env = dict(os.environ)

  random.seed(0)

  # setup a clean environment for each test
  with OpenpilotPrefix():
    prefix = os.environ["OPENPILOT_PREFIX"]

    yield

    # ensure the test doesn't change the prefix
    assert "OPENPILOT_PREFIX" in os.environ and prefix == os.environ["OPENPILOT_PREFIX"]

  os.environ.clear()
  os.environ.update(starting_env)


# If you use setUpClass, the environment variables won't be cleared properly,
# so we need to hook both the function and class pytest fixtures
@pytest.fixture(scope="class", autouse=True)
def openpilot_class_fixture():
  starting_env = dict(os.environ)

  yield

  os.environ.clear()
  os.environ.update(starting_env)


def pytest_collection_modifyitems(config, items):
  skipper = pytest.mark.skip(reason="Skipping tici test on PC")
  for item in items:
    if not TICI and "tici" in item.keywords:
      item.add_marker(skipper)