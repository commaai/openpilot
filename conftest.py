import os
import pytest

from openpilot.common.prefix import OpenpilotPrefix


@pytest.fixture(scope="function", autouse=True)
def global_setup_and_teardown():
  # setup a clean environment for each test
  with OpenpilotPrefix():
    yield


def pytest_collection_modifyitems(config, items):
  # Skip all integration tests unless the environment variable "INTEGRATION" is set
  if os.environ.get("INTEGRATION", None) != "1":
    skipper = pytest.mark.skip(reason="Skipping integration test since INTEGRATION env variable is not set.")
    for item in items:
      if "integration" in item.keywords:
        item.add_marker(skipper)