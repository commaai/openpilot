import os
import pytest

from openpilot.common.prefix import OpenpilotPrefix


@pytest.fixture(scope="function", autouse=True)
def global_setup_and_teardown():
  # setup a clean environment for each test
  with OpenpilotPrefix():
    yield


# we can't mark cpp tests, so add them manually here
ADDITIONAL_EXPLICIT = ["selfdrive/ui/tests/test_translations", "selfdrive/ui/tests/test_sound"]


def pytest_collection_modifyitems(config, items):
  print(config)
  skipper = pytest.mark.skip(reason="Skipping integration test since INTEGRATION env variable is not set.")
  for item in items:
    if os.environ.get("INTEGRATION", None) != "1":
      if "integration" in item.keywords or item.location[0] in ADDITIONAL_EXPLICIT:
        item.add_marker(skipper)