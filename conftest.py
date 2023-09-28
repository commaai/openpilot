import pytest

from openpilot.common.prefix import OpenpilotPrefix


@pytest.fixture(scope="function", autouse=True)
def global_setup_and_teardown():
  # setup a clean environment for each test
  with OpenpilotPrefix():
    yield

# there is currently no way to mark cpp tests, so we have to mark them here
ADDITIONAL_EXPLICIT = ["selfdrive/ui/tests/test_translations", "selfdrive/ui/tests/test_sound"]

def pytest_collection_modifyitems(config, items):
  skipper = pytest.mark.skip(reason="Skipping explicit test since it was not run directly.")
  for item in items:
    test_path = item.location[0]
    is_explicit_test = "explicit" in item.keywords or test_path in ADDITIONAL_EXPLICIT
    was_run_explicitly = test_path in config.option.file_or_dir
    if is_explicit_test and not was_run_explicitly:
      item.add_marker(skipper)